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
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/common.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(softmax)
{
    migraphx::shape s{migraphx::shape::float_type, {10, 1000}};
    migraphx::module m;
    auto x       = m.add_parameter("x", s);
    auto softmax = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    m.add_return({softmax});
    run_pass(m);
    EXPECT(none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "softmax"; }));

    auto reduces = find_all(migraphx::iterator_for(m),
                            [&](auto ins) { return migraphx::contains(ins->name(), "reduce"); });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == 1;
    }));
}

TEST_CASE(reduce_mean)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m;
    auto x           = m.add_parameter("x", s);
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    m.add_return({reduce_mean});
    run_pass(m);
    EXPECT(
        none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "reduce_mean"; }));

    auto reduces = find_all(migraphx::iterator_for(m), [&](auto ins) {
        return migraphx::contains(ins->name(), "reduce_sum");
    });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == -1;
    }));
}

TEST_CASE(reduce_mean_large)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 3, 65536}};
    migraphx::module m;
    auto x           = m.add_parameter("x", s);
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    m.add_return({reduce_mean});
    run_pass(m);
    EXPECT(
        none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "reduce_mean"; }));

    auto reduces = find_all(migraphx::iterator_for(m), [&](auto ins) {
        return migraphx::contains(ins->name(), "reduce_sum");
    });
    EXPECT(all_of(reduces, [](migraphx::instruction_ref ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == -1 and
               ins->get_shape().type() == migraphx::shape::float_type;
    }));
}

TEST_CASE(reduce_mean_accuracy)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4.f, 13.f, 22.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3, 3}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4.f, 13.f, 22.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3, 3}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{3.f, 4.f, 5.f, 12.f, 13.f, 14.f, 21.f, 22.f, 23.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::int32_type, {1, 3, 2, 2}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold{1, 2, 5, 6, 9, 10};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy5)
{
    using migraphx::half;
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::half_type, {1, 3, 2, 2}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<half> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<half> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{half{1}, half{2}, half{5}, half{6}, half{9}, half{10}};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

static migraphx::instruction_ref
add_reduce_mean(migraphx::module& m, std::vector<std::size_t> axes, migraphx::instruction_ref input)
{
    auto reduce_size = migraphx::transform_accumulate(
        axes.begin(), axes.end(), std::size_t{1}, std::multiplies<>{}, [&](auto axis) {
            return input->get_shape().lens()[axis];
        });
    auto t      = input->get_shape().type();
    auto rl     = m.add_literal(migraphx::literal{{t, {1}}, {reduce_size}});
    auto div    = migraphx::add_common_op(m, migraphx::make_op("div"), {input, rl});
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), div);
    return m.add_instruction(migraphx::make_op("convert", {{"target_type", t}}), reduce);
}

TEST_CASE(reduce_mean_variance_sqdiff)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), sqdiff);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_mul_x_minus)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sub      = m1.add_instruction(migraphx::make_op("sub"), x, meanb);
        auto mul      = m1.add_instruction(migraphx::make_op("mul"), sub, sub);
        auto variance = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), mul);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_pow_x_minus)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sub      = m1.add_instruction(migraphx::make_op("sub"), x, meanb);
        auto two      = m1.add_literal(migraphx::literal{migraphx::shape::float_type, {2}});
        auto pow      = migraphx::add_common_op(m1, migraphx::make_op("pow"), {sub, two});
        auto variance = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), pow);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_diff_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto mean1  = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto mean2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean1);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), y, mean2b);
        auto mean2  = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), sqdiff);
        m1.add_return({mean1, mean2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto mean1  = add_reduce_mean(m2, {2}, x);
        auto mean2b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean1);
        auto sqdiff = m2.add_instruction(migraphx::make_op("sqdiff"), y, mean2b);
        auto mean2  = add_reduce_mean(m2, {2}, sqdiff);
        m2.add_return({mean1, mean2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_sqdiff_diff_axes)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), sqdiff);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto mean = add_reduce_mean(m2, {2}, x);
        auto meanb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff   = m2.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance = add_reduce_mean(m2, {0, 2}, sqdiff);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
