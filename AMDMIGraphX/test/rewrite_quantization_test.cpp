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
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/env.hpp>

#include <migraphx/serialize.hpp>
#include <migraphx/pass_manager.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK_WORKAROUNDS);

static bool is_quantizelinear(migraphx::instruction& ins) { return ins.name() == "quantizelinear"; }
static bool is_dequantizelinear(migraphx::instruction& ins)
{
    return ins.name() == "dequantizelinear";
}
static bool is_clip_scalar(migraphx::instruction& ins)
{
    if(ins.name() == "clip")
    {
        assert(ins.inputs().size() > 1);
        return (std::all_of(ins.inputs().begin() + 1, ins.inputs().end(), [](auto input) {
            return input->get_shape().scalar();
        }));
    }
    return false;
}

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_quantization{}});
}

static migraphx::argument eval(const migraphx::program& p)
{
    auto r = p.eval({});
    EXPECT(r.size() == 1);
    return r.front();
}

TEST_CASE(quantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {-300, 200, 129, 1, 2, 3, 500, 1000, 50};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    run_pass(*p2.get_main_module());
    EXPECT(eval(p1) == eval(p2));
    EXPECT(any_of(*p1.get_main_module(), &is_quantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_quantizelinear));
    // ensure clip literals created in quantized program are scalar
    // unless CK workarounds are enabled
    if(migraphx::enabled(MIGRAPHX_ENABLE_CK_WORKAROUNDS{}))
        EXPECT(none_of(*p2.get_main_module(), &is_clip_scalar));
    else
        EXPECT(any_of(*p2.get_main_module(), &is_clip_scalar));
}

TEST_CASE(dequantizelinear)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> xv = {0, 1, 2, 5, 10, 50, 100, 150, 250};
    migraphx::shape ss{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    migraphx::shape zs{migraphx::shape::float_type, {1, 3, 3}};
    std::vector<float> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto create_program   = [&]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, s, z);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    run_pass(*p2.get_main_module());
    EXPECT(eval(p1) == eval(p2));
    EXPECT(any_of(*p1.get_main_module(), &is_dequantizelinear));
    EXPECT(none_of(*p2.get_main_module(), &is_dequantizelinear));
}

// has a nearbyint operation
TEST_CASE(quantize_to_integral_type)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> xv = {
        -300, 600, 129, -1000, 4, 3, -6, 600, 550, -300, 600, 129, -1000, 4, 3, -6, 600, 550};
    migraphx::shape ss{migraphx::shape::float_type, {2, 3, 3}};
    std::vector<float> sv = {2, 2, 2, 4, 4, 4, 6, 6, 6, 2, 2, 2, 4, 4, 4, 6, 6, 6};
    migraphx::shape zs{migraphx::shape::int8_type, {2, 3, 3}};
    std::vector<uint8_t> zv = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    migraphx::program p_run;
    {
        auto* mm = p_run.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(ss, sv);
        auto z   = mm->add_literal(zs, zv);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
    };

    migraphx::program p_expected;
    {
        auto* mm        = p_expected.get_main_module();
        auto x          = mm->add_literal(xs, xv);
        auto s          = mm->add_literal(ss, sv);
        auto z          = mm->add_literal(zs, zv);
        auto div        = mm->add_instruction(migraphx::make_op("div"), x, s);
        auto nearby_int = mm->add_instruction(migraphx::make_op("nearbyint"), div);
        auto zero_point = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), z);
        auto add_zero_point = mm->add_instruction(migraphx::make_op("add"), nearby_int, zero_point);
        double max_quant    = 0;
        double min_quant    = 0;
        auto zp_shape       = add_zero_point->get_shape();
        zs.visit_type([&](auto qt) {
            max_quant = qt.max();
            min_quant = qt.min();
        });
        auto min_arg =
            mm->add_literal(migraphx::literal{migraphx::shape{zp_shape.type()}, {min_quant}});
        auto max_arg =
            mm->add_literal(migraphx::literal{migraphx::shape{zp_shape.type()}, {max_quant}});
        min_arg = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", zp_shape.lens()}}), min_arg);
        max_arg = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", zp_shape.lens()}}), max_arg);
        auto saturate =
            mm->add_instruction(migraphx::make_op("clip"), {add_zero_point, min_arg, max_arg});
        mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), saturate);
    };

    run_pass(*p_run.get_main_module());
    EXPECT(p_run == p_expected);
}

// should not have a nearbyint operation
TEST_CASE(quantize_to_floating_point_type)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 2, 2}};
    migraphx::shape zs{migraphx::shape::get_type<migraphx::fp8::fp8e4m3fn>{}, {2, 2, 2}};
    std::vector<float> xv  = {0.5, 0.75, -0.4375, 0.6875, -0.9375, -0.9375, 0.625, -0.5625};
    std::vector<float> sv  = {0.25, 0.75, 0.5625, 0.4375, 0.8125, -0.6875, 0.875, -0.0625};
    std::vector<float> tmp = {0.6875, 0.75, -0.75, 0.5, -0.0625, 0.0625, -0.375, 0.25};
    std::vector<migraphx::fp8::fp8e4m3fn> zero_pts;
    std::transform(tmp.begin(), tmp.end(), std::back_inserter(zero_pts), [](auto x) {
        return migraphx::fp8::fp8e4m3fn(x);
    });

    migraphx::program p_run;
    {
        auto* mm = p_run.get_main_module();
        auto x   = mm->add_literal(xs, xv);
        auto s   = mm->add_literal(xs, sv);
        auto z   = mm->add_literal(zs, zero_pts);
        mm->add_instruction(migraphx::make_op("quantizelinear"), x, s, z);
    };

    migraphx::program p_expected;
    {
        auto* mm        = p_expected.get_main_module();
        auto x          = mm->add_literal(xs, xv);
        auto s          = mm->add_literal(xs, sv);
        auto z          = mm->add_literal(zs, zero_pts);
        auto div        = mm->add_instruction(migraphx::make_op("div"), x, s);
        auto zero_point = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), z);
        auto add_zero_point = mm->add_instruction(migraphx::make_op("add"), div, zero_point);
        double max_quant    = 0;
        double min_quant    = 0;
        auto zp_shape       = add_zero_point->get_shape();
        zs.visit_type([&](auto qt) {
            max_quant = qt.max();
            min_quant = qt.min();
        });
        auto min_arg =
            mm->add_literal(migraphx::literal{migraphx::shape{zp_shape.type()}, {min_quant}});
        auto max_arg =
            mm->add_literal(migraphx::literal{migraphx::shape{zp_shape.type()}, {max_quant}});
        min_arg = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", zp_shape.lens()}}), min_arg);
        max_arg = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", zp_shape.lens()}}), max_arg);
        auto saturate =
            mm->add_instruction(migraphx::make_op("clip"), {add_zero_point, min_arg, max_arg});
        mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fn_type}}),
            saturate);
    };

    run_pass(*p_run.get_main_module());
    EXPECT(p_run == p_expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
