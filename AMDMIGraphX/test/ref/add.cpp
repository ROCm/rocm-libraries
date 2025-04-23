/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(add_broadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 2}};
    std::vector<float> b_data{0, -1, -2, -3};
    uint64_t axis = 0;
    auto l1       = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2       = mm->add_literal(migraphx::literal{b_shape, b_data});
    auto l3       = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l1->get_shape().lens()}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l1, l3);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    EXPECT(result.get_shape().packed());
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(add_multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> a_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 1}};
    std::vector<float> b_data{0, -1, -2, -3};
    auto l1 = mm->add_literal(migraphx::literal{a_shape, a_data});
    auto l2 = mm->add_literal(migraphx::literal{b_shape, b_data});
    auto l3 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l1);
    auto l4 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l2);
    mm->add_instruction(migraphx::make_op("add"), l3, l4);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    EXPECT(result.get_shape().packed());
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l1 = mm->add_literal(migraphx::literal{s, {-1, 0, 1}});
    auto l2 = mm->add_literal(migraphx::literal{s, {1, 2, 3}});
    mm->add_instruction(migraphx::make_op("add"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(add_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("add"), x, y);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{-1, 0, 1};
    std::vector<float> y_data{1, 2, 3};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, x_data.data());
    params0["y"] = migraphx::argument(input_fixed_shape0, y_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 2, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(fp16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {1}};
    migraphx::half a{1.5};
    migraphx::half b{2.5};
    migraphx::half c{4.0};
    auto l0 = mm->add_literal(migraphx::literal{s, {a}});
    auto l1 = mm->add_literal(migraphx::literal{s, {b}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::half> results_vector(1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{c};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(bf16_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bf16_type, {1}};
    migraphx::bf16 a{1.5};
    migraphx::bf16 b{2.5};
    migraphx::bf16 c{4.0};
    auto l0 = mm->add_literal(migraphx::literal{s, {a}});
    auto l1 = mm->add_literal(migraphx::literal{s, {b}});
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<migraphx::bf16> results_vector(1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::bf16> gold{c};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(fp32_fp16_test)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = mm->add_literal(migraphx::literal(s, data));
        auto l2 = mm->add_literal(migraphx::literal(s, data));
        mm->add_instruction(migraphx::make_op("add"), l1, l2);
        return p;
    };

    auto test_case = [&](const std::vector<std::string>& op_names) {
        std::vector<float> gold_res = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
        auto p                      = create_program();
        migraphx::quantize_fp16(p, op_names);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> res;
        result.visit([&](auto output) { res.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_rms_range(res, gold_res));
    };

    test_case({"all"});
    test_case({"add"});
}
