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
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/optional.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct concat
{
    concat(std::size_t axis) { op.axis = axis; }
    migraphx::op::concat op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axis"] = migraphx::value::array{migraphx::op::normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "eliminate_concat::concat"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        inputs.pop_back();
        return op.normalize_compute_shape(std::move(inputs));
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

struct concat_test_optimization
{
    /// A unique name used to identify the allocate operator
    std::string allocate() const { return "allocate"; }
    /// Return the lowered concat operator
    std::optional<migraphx::op::concat> get_concat(const migraphx::operation& op) const
    {
        if(op.name() != "eliminate_concat::concat")
            return std::nullopt;
        return migraphx::any_cast<concat>(op).op;
    }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::eliminate_concat{concat_test_optimization{}},
                          migraphx::dead_code_elimination{}});
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

struct simple_op
{
    std::string name() const { return "simple_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(1);
        return inputs.at(0);
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& args) const
    {
        return args.at(0);
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

template <class... Ts>
static migraphx::shape create_shape(Ts... xs)
{
    return migraphx::shape{migraphx::shape::float_type, {std::size_t(xs)...}};
}

using load     = migraphx::op::load;
using identity = migraphx::op::identity;

TEST_CASE(simple)
{
    auto create_test_program = [] {
        migraphx::module m;

        auto a1          = m.add_instruction(allocate{create_shape(1)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(1)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = m.add_instruction(allocate{create_shape(2)});
        m.add_instruction(concat(axis), m1, m2, a3);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;

        auto a1 = m.add_instruction(allocate{create_shape(2)});
        auto l1 = m.add_instruction(load{create_shape(1), 0}, a1);
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(load{create_shape(1), 4}, a1);
        auto m2 = m.add_instruction(simple_op{}, l2);
        m.add_instruction(identity{}, a1, m1, m2);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(negative_axis1)
{
    auto create_test_program = [] {
        migraphx::module m;

        auto a1          = m.add_instruction(allocate{create_shape(2, 2)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(2, 2)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = -1;
        auto a3          = m.add_instruction(allocate{create_shape(4, 2)});
        m.add_instruction(concat(axis), m1, m2, a3);
        return m;
    };
    auto create_control_program = create_test_program;

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(negative_axis2)
{
    auto create_test_program = [] {
        migraphx::module m;

        auto a1          = m.add_instruction(allocate{create_shape(2, 2)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(2, 2)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = -2;
        auto a3          = m.add_instruction(allocate{create_shape(4, 2)});
        m.add_instruction(concat(axis), m1, m2, a3);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;

        auto a1 = m.add_instruction(allocate{create_shape(4, 2)});
        auto l1 = m.add_instruction(load{create_shape(2, 2), 0}, a1);
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(load{create_shape(2, 2), 16}, a1);
        auto m2 = m.add_instruction(simple_op{}, l2);
        m.add_instruction(identity{}, a1, m1, m2);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(negative_axis3)
{
    auto create_test_program = [] {
        migraphx::module m;

        auto a1          = m.add_instruction(allocate{create_shape(1, 2, 2)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(1, 2, 2)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = -2;
        auto a3          = m.add_instruction(allocate{create_shape(1, 4, 2)});
        m.add_instruction(concat(axis), m1, m2, a3);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;

        auto a1 = m.add_instruction(allocate{create_shape(1, 4, 2)});
        auto l1 = m.add_instruction(load{create_shape(1, 2, 2), 0}, a1);
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(load{create_shape(1, 2, 2), 16}, a1);
        auto m2 = m.add_instruction(simple_op{}, l2);
        m.add_instruction(identity{}, a1, m1, m2);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(reversed)
{
    auto create_test_program = [] {
        migraphx::module m;

        auto a1          = m.add_instruction(allocate{create_shape(1)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(1)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = m.add_instruction(allocate{create_shape(2)});
        m.add_instruction(concat(axis), m2, m1, a3);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;

        auto a1 = m.add_instruction(allocate{create_shape(2)});
        auto l1 = m.add_instruction(load{create_shape(1), 4}, a1);
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(load{create_shape(1), 0}, a1);
        auto m2 = m.add_instruction(simple_op{}, l2);
        m.add_instruction(identity{}, a1, m2, m1);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(nested)
{
    auto concat_test_program = [](auto& m) {
        auto a1          = m.add_instruction(allocate{create_shape(1)});
        auto m1          = m.add_instruction(simple_op{}, a1);
        auto a2          = m.add_instruction(allocate{create_shape(1)});
        auto m2          = m.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = m.add_instruction(allocate{create_shape(2)});
        return m.add_instruction(concat(axis), m1, m2, a3);
    };
    auto create_test_program = [&] {
        migraphx::module m;
        auto concat1     = concat_test_program(m);
        auto concat2     = concat_test_program(m);
        std::size_t axis = 0;
        auto a1          = m.add_instruction(allocate{create_shape(4)});
        m.add_instruction(concat(axis), concat1, concat2, a1);
        return m;
    };
    auto concat_control_program = [](auto& m, auto a1) {
        auto l1 = m.add_instruction(load{create_shape(1), 0}, a1);
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(load{create_shape(1), 4}, a1);
        auto m2 = m.add_instruction(simple_op{}, l2);
        return m.add_instruction(identity{}, a1, m1, m2);
    };
    auto create_control_program = [&] {
        migraphx::module m;
        auto a1      = m.add_instruction(allocate{create_shape(4)});
        auto l1      = m.add_instruction(load{create_shape(2), 0}, a1);
        auto concat1 = concat_control_program(m, l1);
        auto l2      = m.add_instruction(load{create_shape(2), 8}, a1);
        auto concat2 = concat_control_program(m, l2);
        m.add_instruction(identity{}, a1, concat1, concat2);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(basic)
{
    auto create_test_program = [] {
        migraphx::module m;
        auto a1 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}});
        auto m1 = m.add_instruction(simple_op{}, a1);
        auto a2 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}});
        auto m2 = m.add_instruction(simple_op{}, a2);
        auto a3 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}});
        auto p3          = m.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = m.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        m.add_instruction(concat(axis), m1, m2, p3, a4);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;
        auto a1 = m.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        auto l1 = m.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}, 0}, {a1});
        auto m1 = m.add_instruction(simple_op{}, l1);
        auto l2 = m.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}, 512}, {a1});
        auto m2 = m.add_instruction(simple_op{}, l2);
        auto l3 = m.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}, 1280}, {a1});
        auto p3 = m.add_instruction(simple_op{}, l3);
        m.add_instruction(identity{}, {a1, m1, m2, p3});
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(wont_work)
{
    auto create_test_program = [] {
        migraphx::module m;
        auto a1 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto m1 = m.add_instruction(simple_op{}, a1);
        auto a2 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto m2 = m.add_instruction(simple_op{}, a2);
        auto a3 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = m.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = m.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        m.add_instruction(concat(axis), m1, m2, p3, a4);
        return m;
    };
    auto create_control_program = [] {
        migraphx::module m;
        auto a1 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto m1 = m.add_instruction(simple_op{}, a1);
        auto a2 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto m2 = m.add_instruction(simple_op{}, a2);
        auto a3 =
            m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = m.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = m.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        m.add_instruction(concat(axis), m1, m2, p3, a4);
        return m;
    };

    auto m1 = create_test_program();
    auto m2 = create_control_program();
    run_pass(m1);

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
