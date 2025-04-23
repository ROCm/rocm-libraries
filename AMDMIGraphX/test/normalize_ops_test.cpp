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
#include <migraphx/normalize_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct normalize_test_op
{
    std::vector<int64_t> axes = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"));
    }

    migraphx::value attributes() const
    {
        migraphx::value normalize;
        normalize["axes"] = migraphx::value::array{migraphx::op::normalize_attribute::clip_max,
                                                   migraphx::op::normalize_attribute::clip_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "normalize_ops_test::test_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return migraphx::argument{output_shape};
    }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::normalize_ops{}, migraphx::dead_code_elimination{}});
}

static migraphx::module create_gather(int64_t axis)
{
    migraphx::module m;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape si{migraphx::shape::int64_type, {2, 3}};
    auto di = m.add_parameter("data", sd);
    auto ii = m.add_parameter("ind", si);
    auto r  = m.add_instruction(migraphx::make_op("gather", {{"axis", axis}}), di, ii);
    m.add_return({r});

    return m;
}

TEST_CASE(gather_test)
{

    auto m1 = create_gather(-3);
    auto m2 = create_gather(0);
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(gather_test_1)
{
    auto m1 = create_gather(1);
    auto m2 = create_gather(1);
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_padded_op(const std::vector<size_t>& pad_vals)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(migraphx::make_op("pooling", {{"padding", pad_vals}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(padding_attr_test)
{
    migraphx::module m1 = create_padded_op({0, 1});
    migraphx::module m2 = create_padded_op({0, 1, 0, 1});
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_reduce_mean(const std::vector<int64_t>& axes)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(reduce_mean_test)
{
    migraphx::module m1 = create_reduce_mean({0, 1, -1});
    migraphx::module m2 = create_reduce_mean({0, 1, 3});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(reduce_mean_test_1)
{
    migraphx::module m1 = create_reduce_mean({0, 1, 2});
    migraphx::module m2 = create_reduce_mean({0, 1, 2});
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_slice(const std::vector<int64_t>& axes,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t>& ends)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    auto si = m.add_parameter("data", s);
    auto r  = m.add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}), si);
    m.add_return({r});

    return m;
}

TEST_CASE(slice_test)
{
    migraphx::module m1 = create_slice({0, 1, -1}, {-5, 1, -3}, {2, 2, 8});
    migraphx::module m2 = create_slice({0, 1, 3}, {0, 1, 2}, {2, 2, 5});
    run_pass(m1);

    EXPECT(m1 == m2);
}

TEST_CASE(slice_test_1)
{
    migraphx::module m1 = create_slice({0, 1, 3}, {0, 1, -3}, {1, 2, 5});
    migraphx::module m2 = create_slice({0, 1, 3}, {0, 1, 2}, {1, 2, 5});
    run_pass(m1);

    EXPECT(m1 == m2);
}

static migraphx::module create_test_op(const std::vector<int64_t>& axes)
{
    migraphx::module m;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    auto di = m.add_parameter("data", sd);
    auto r  = m.add_instruction(normalize_test_op{axes}, di);
    m.add_return({r});

    return m;
}

TEST_CASE(test_op)
{
    std::vector<int64_t> axes1 = {-4, 5};
    auto m1                    = create_test_op(axes1);

    std::vector<int64_t> axes2 = {1, 2};
    auto m2                    = create_test_op(axes2);

    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
