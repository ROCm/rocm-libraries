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
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m, std::size_t align = 32)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_allocation{"allocate", align}, migraphx::dead_code_elimination{}});
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

TEST_CASE(basic)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {8}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {40}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (8 * 4 + 40 * 4 + 200 * 4));
}

TEST_CASE(aligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (32 + 32 + 200 * 4));
}

TEST_CASE(unaligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m, 1);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

TEST_CASE(float_aligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m, 4);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
