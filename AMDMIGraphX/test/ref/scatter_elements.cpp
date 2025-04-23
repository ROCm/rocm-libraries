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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

static migraphx::program create_scatter_elements_program(const std::string& reduction_mode,
                                                         int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 0.0f);

    migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> vi = {1, 0, 2, 0, 2, 1};

    migraphx::shape su{migraphx::shape::float_type, {2, 3}};
    std::vector<float> vu = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});

    auto r = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}

TEST_CASE(scatter_elements_axis_0_test)
{
    migraphx::program p = create_scatter_elements_program("none", 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatter_elements_axis_neg_2_test)
{
    migraphx::program p = create_scatter_elements_program("none", -2);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatter_elements_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program("none", 1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.1, 1.0, 1.2, 2.0, 2.2, 2.1, 0.0, 0.0, 0.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

static migraphx::program create_scatter_elements_program2(const std::string& reduction_mode,
                                                          int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {1, 5}};
    std::vector<float> vd({1., 2., 3., 4., 5.});

    migraphx::shape si{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> vi = {1, 3};

    migraphx::shape su{migraphx::shape::float_type, {1, 2}};
    std::vector<float> vu = {1.1, 2.1};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}

TEST_CASE(scatter_elements_none_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program2("none", 1);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_none = {1.0, 1.1, 3.0, 2.1, 5.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_none));
}

TEST_CASE(scatter_elements_mul_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program2("mul", 1);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_mul = {1.0, 2.2, 3.0, 8.4, 5.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_mul));
}

TEST_CASE(scatter_elements_add_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program2("add", 1);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_add = {1.0, 3.1, 3.0, 6.1, 5.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_add));
}

TEST_CASE(scatter_elements_min_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program2("min", 1);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_min = {1.0, 1.1, 3.0, 2.1, 5.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_min));
}

TEST_CASE(scatter_elements_max_axis_1_test)
{
    migraphx::program p = create_scatter_elements_program2("max", 1);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_max = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_max));
}

static auto scatter_elements_duplicate_index_test(const std::string& reduction_mode)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {1, 5}};
    std::vector<float> vd({1., 2., 3., 4., 5.});

    migraphx::shape si{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> vi = {1, 1};

    migraphx::shape su{migraphx::shape::float_type, {1, 2}};
    std::vector<float> vu = {1.1, 2.1};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction_mode, {{"axis", 1}}), ld, li, lu);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    return results_vector;
}

TEST_CASE(scatter_elements_add_axis_1_duplicate_idx_test)
{
    const auto results = scatter_elements_duplicate_index_test("add");
    const std::vector<float> gold{1.0, 5.2, 3.0, 4.0, 5.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_mul_axis_1_duplicate_idx_test)
{
    const auto results = scatter_elements_duplicate_index_test("mul");
    const std::vector<float> gold{1.0, 4.62, 3.0, 4.0, 5.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_min_axis_1_duplicate_idx_test)
{
    const auto results = scatter_elements_duplicate_index_test("min");
    const std::vector<float> gold{1.0, 1.1, 3.0, 4.0, 5.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_max_axis_1_duplicate_idx_test)
{
    const auto results = scatter_elements_duplicate_index_test("max");
    const std::vector<float> gold{1.0, 2.1, 3.0, 4.0, 5.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

static migraphx::program create_scatter_elements_program_3x3(const std::string& reduction_mode,
                                                             int axis)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 3.0f);

    migraphx::shape si{migraphx::shape::int32_type, {3, 2}};
    std::vector<int> vi = {1, 0, 0, 2, 2, 1};

    migraphx::shape su{migraphx::shape::float_type, {3, 2}};
    std::vector<float> vu = {1.0, 7.0, 1.1, 7.1, 1.2, 7.2};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction_mode, {{"axis", axis}}), ld, li, lu);
    mm->add_return({r});
    return p;
}

TEST_CASE(scatter_elements_none_axis_0_3x3_test)
{
    migraphx::program p = create_scatter_elements_program_3x3("none", 0);
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_none2 = {1.1, 7.0, 3.0, 1.0, 7.2, 3.0, 1.2, 7.1, 3.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_none2));
}

TEST_CASE(scatter_elements_add_axis_0_3x3_test)
{
    migraphx::program p = create_scatter_elements_program_3x3("add", 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_a3 = {4.1, 10.0, 3.0, 4.0, 10.2, 3.0, 4.2, 10.1, 3.0};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_a3));
}

TEST_CASE(scatter_elements_mul_axis_0_3x3_test)
{
    migraphx::program p = create_scatter_elements_program_3x3("mul", 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_mul2 = {3.3, 21.0, 3.0, 3.0, 21.6, 3.0, 3.6, 21.3, 3.0};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_mul2));
}

TEST_CASE(scatter_elements_min_axis_0_3x3_test)
{
    migraphx::program p = create_scatter_elements_program_3x3("min", 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_min = {1.1, 3.0, 3.0, 1.0, 3.0, 3.0, 1.2, 3.0, 3.0};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_min));
}

TEST_CASE(scatter_elements_max_axis_0_3x3_test)
{
    migraphx::program p = create_scatter_elements_program_3x3("max", 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_max = {3.0, 7.0, 3.0, 3.0, 7.2, 3.0, 3.0, 7.1, 3.0};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_max));
}

static auto scatter_elements_3x3_duplicate_index_test(const std::string& reduction_mode)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 1.0f);

    migraphx::shape si{migraphx::shape::int32_type, {3, 2}};
    std::vector<int> vi = {1, 0, 1, 0, 1, 0};

    migraphx::shape su{migraphx::shape::float_type, {3, 2}};
    std::vector<float> vu = {0.9, 2.0, 1.1, 2.1, 1.2, 2.2};

    auto ld = mm->add_literal(migraphx::literal{sd, vd});
    auto li = mm->add_literal(migraphx::literal{si, vi});
    auto lu = mm->add_literal(migraphx::literal{su, vu});
    auto r  = mm->add_instruction(
        migraphx::make_op("scatter_" + reduction_mode, {{"axis", 0}}), ld, li, lu);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));

    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    return results_vector;
}

TEST_CASE(scatter_elements_skip_out_of_bounds_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vd(sd.elements(), 0.0f);

    migraphx::shape si{migraphx::shape::int32_type, {2, 3}};
    // clang-format off
    std::vector<int> vi = {
        -1, 0, 20,
        0, -8, 1
    };
    // clang-format on

    migraphx::shape su{migraphx::shape::float_type, {2, 3}};
    // clang-format off
    std::vector<float> vu = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    // clang-format on

    auto lit_data    = mm->add_literal(migraphx::literal{sd, vd});
    auto lit_inds    = mm->add_literal(migraphx::literal{si, vi});
    auto lit_updates = mm->add_literal(migraphx::literal{su, vu});

    auto r = mm->add_instruction(
        migraphx::make_op("scatter_none", {{"axis", 0}, {"skip_out_of_bounds", true}}),
        lit_data,
        lit_inds,
        lit_updates);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // clang-format off
    std::vector<float> gold = {
        4.0, 2.0, 0.0,
        0.0, 0.0, 6.0,
        1.0, 0.0, 0.0
    };
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(scatter_elements_add_axis_0_3x3_duplicate_index_test)
{
    const auto results            = scatter_elements_3x3_duplicate_index_test("add");
    const std::vector<float> gold = {1.0, 7.3, 1.0, 4.2, 1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_mul_axis_0_3x3_duplicate_index_test)
{
    const auto results            = scatter_elements_3x3_duplicate_index_test("mul");
    const std::vector<float> gold = {1.0, 9.24, 1.0, 1.188, 1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_min_axis_0_3x3_duplicate_index_test)
{
    const auto results            = scatter_elements_3x3_duplicate_index_test("min");
    const std::vector<float> gold = {1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(scatter_elements_max_axis_0_3x3_duplicate_index_test)
{
    const auto results            = scatter_elements_3x3_duplicate_index_test("max");
    const std::vector<float> gold = {1.0, 2.2, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}
