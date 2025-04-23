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

TEST_CASE(slice_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                        l0);
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
    auto result           = p.eval({}).back();
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
        l0);
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_var_inputs_static0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), l0, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {1};
    std::vector<int32_t> end_data   = {3};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), l0, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {-2};
    std::vector<int32_t> end_data   = {2831};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::float_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {3}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    auto axes   = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice"), l0, starts, ends, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> start_data = {0, 0, 0};
    std::vector<int64_t> end_data   = {2, 2, 2};
    std::vector<int64_t> axes_data  = {0, 1, 2};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    params["axes"]                  = migraphx::argument(s1, axes_data.data());
    auto result                     = p.eval(params).back();
    std::vector<float> gold         = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<float> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"ends", {10}}}), input, starts);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> start_data = {1};
    params["input"]             = migraphx::argument(s2, input_data.data());
    params["starts"]            = migraphx::argument(s1, start_data.data());
    auto result                 = p.eval(params).back();
    std::vector<int> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {-5}}}), input, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> ends_data = {3};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["ends"]             = migraphx::argument(s1, ends_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto axes = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"starts", {1}}, {"ends", {-1}}}), input, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> axes_data = {2};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["axes"]             = migraphx::argument(s1, axes_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {1, 4, 7, 10};
    std::vector<int> results_vector(2 * 2 * 1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), input, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> starts_data = {1};
    std::vector<int> ends_data   = {std::numeric_limits<int>::max()};
    params["input"]              = migraphx::argument(s2, input_data.data());
    params["starts"]             = migraphx::argument(s1, starts_data.data());
    params["ends"]               = migraphx::argument(s1, ends_data.data());
    auto result                  = p.eval(params).back();
    std::vector<int> gold        = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto axes   = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"ends", {std::numeric_limits<int>::max()}}}),
                        input,
                        starts,
                        axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> starts_data = {1};
    std::vector<int> axes_data   = {2};
    params["input"]              = migraphx::argument(s2, input_data.data());
    params["starts"]             = migraphx::argument(s1, starts_data.data());
    params["axes"]               = migraphx::argument(s1, axes_data.data());
    auto result                  = p.eval(params).back();
    std::vector<int> gold        = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn5)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    auto axes = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"starts", {-4}}}), input, ends, axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> ends_data = {2};
    std::vector<int> axes_data = {2};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["ends"]             = migraphx::argument(s1, ends_data.data());
    params["axes"]             = migraphx::argument(s1, axes_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn6)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), input, starts, ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> start_data = {1};
    std::vector<int> end_data   = {3};
    params["input"]             = migraphx::argument(s2, input_data.data());
    params["starts"]            = migraphx::argument(s1, start_data.data());
    params["ends"]              = migraphx::argument(s1, end_data.data());
    auto result                 = p.eval(params).back();
    std::vector<int> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_dyn_test0)
{
    // Slice a single dynamic dimension. ax1 slice limits are smaller than min; ax2 "ends" is
    // too large
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 3}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1, 2}}, {"starts", {0, 1}}, {"ends", {1, 6}}}), x);
    migraphx::shape s2{migraphx::shape::int32_type, {{2, 3}, {1, 1}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));

    //  the strides of sresult are those of the original shape, not
    // reduced to sliced size.
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 1, 2}, {6, 3, 1}};
    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 2, 7, 8};
    std::vector<int> results_vector(2 * 1 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_dyn_test1)
{
    // Slice all three dynamic dimensions
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
        x);

    migraphx::shape s2{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}
