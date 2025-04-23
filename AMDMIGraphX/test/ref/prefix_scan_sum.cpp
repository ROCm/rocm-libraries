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
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(prefix_scan_sum_1d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 10.0, 15.0, 21.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_dyn_1d)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{5, 8}};
    migraphx::shape s{migraphx::shape::float_type, dd};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {6}};
    migraphx::parameter_map params0;
    params0["X"] = migraphx::argument(input_fixed_shape0, a.data());

    auto result = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 10.0, 15.0, 21.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_2d_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_2d_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_3d_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_3d_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_3d_3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 2}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_exclusive_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {8}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.0, 1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 16.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_exclusive_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", true}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_exclusive_reverse)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {6}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 5, 6}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}, {"reverse", true}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{20.0, 18.0, 15.0, 11.0, 6.0, 0.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_negative_axis_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", -3}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_negative_axis_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", -2}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_negative_axis_3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 3}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", -1}, {"exclusive", false}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_reverse_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {8}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum",
                                          {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{20.0, 19.0, 17.0, 14.0, 10.0, 9.0, 7.0, 4.0};
    EXPECT(results_vector == gold);
}

TEST_CASE(prefix_scan_sum_reverse_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 4, 1, 2, 3, 4}};
    auto l0    = mm->add_literal(input);
    mm->add_instruction(migraphx::make_op("prefix_scan_sum",
                                          {{"axis", 0}, {"exclusive", false}, {"reverse", true}}),
                        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0};
    EXPECT(results_vector == gold);
}
