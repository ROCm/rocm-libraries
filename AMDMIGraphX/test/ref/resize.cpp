/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(resize_test_1)
{
    // batch size 1, 1 color channel, resize 3x3 to 5x8
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    // non-matching sizes/scales attributes are ignored for 2 input arguments
    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {1}},
                                           {"scales", {1}},
                                           {"nearest_mode", "floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0,
                        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f,
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f, 
        6.5f, 6.5f, 6.5f, 6.5f, 7.5f, 7.5f, 7.5f, 8.5f
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_test_2)
{
    // nearest_mode= "round_prefer_floor" coordinate_transformation_mode= "pytorch_half_pixel"
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    mm->add_instruction(
        migraphx::make_op("resize",
                          {{"nearest_mode", "round_prefer_floor"},
                           {"coordinate_transformation_mode", "pytorch_half_pixel"}}),
        a0,
        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f,  0.5f,  0.5f,  1.5f,  1.5f,  2.5f,  2.5f,  2.5f,
        0.5f,  0.5f,  0.5f,  1.5f,  1.5f,  2.5f,  2.5f,  2.5f,
        3.5f,  3.5f,  3.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,
        6.5f,  6.5f,  6.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,
        6.5f,  6.5f,  6.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_test_3)
{
    // nearest_mode= "ceil" coordinate_transformation_mode= "pytorch_half_pixel"
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    mm->add_instruction(migraphx::make_op("resize",
                                          {{"nearest_mode", "ceil"},
                                           {"coordinate_transformation_mode", "align_corners"}}),
                        a0,
                        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f,  1.5f,  1.5f,  1.5f,  2.5f,  2.5f,  2.5f,  2.5f,
        3.5f,  4.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,  5.5f,
        3.5f,  4.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,  5.5f,
        6.5f,  7.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,
        6.5f,  7.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_test_4)
{
    // nearest_mode= "ceil" coordinate_transformation_mode= "pytorch_half_pixel"
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    mm->add_instruction(
        migraphx::make_op(
            "resize", {{"nearest_mode", "ceil"}, {"coordinate_transformation_mode", "asymmetric"}}),
        a0,
        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f,  1.5f,  1.5f,  2.5f,  2.5f,  2.5f,  2.5f,  2.5f,
        3.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,  5.5f,  5.5f,
        6.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f,
        6.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f,
        6.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_test_5)
{
    // nearest_mode= "ceil" coordinate_transformation_mode= "tf_half_pixel_for_nn"
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    mm->add_instruction(
        migraphx::make_op(
            "resize",
            {{"nearest_mode", "ceil"}, {"coordinate_transformation_mode", "tf_half_pixel_for_nn"}}),
        a0,
        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        4.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,  5.5f,  5.5f,
        4.5f,  4.5f,  4.5f,  5.5f,  5.5f,  5.5f,  5.5f,  5.5f,
        7.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f,
        7.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f,
        7.5f,  7.5f,  7.5f,  8.5f,  8.5f,  8.5f,  8.5f,  8.5f
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_upsample_test_2)
{
    // batch size 2, 1 color channel, resize 3x5 by 1.6x
    // same input/output as resize_upsample_f_dyn_test
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(2 * 3 * 5);
    std::iota(data.begin(), data.end(), 0.1);
    // should upscale to 2x1x4x8
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3, 5}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    //   scale input
    migraphx::shape scale_input{migraphx::shape::float_type, {4}};
    std::vector<float> scale_values = {1.0, 1.0, 1.601, 1.601};
    auto a1                         = mm->add_literal(migraphx::literal{scale_input, scale_values});

    // a0 = input data
    // a1 = scales
    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {}},
                                           {"scales", {1}},
                                           {"nearest_mode", "round_prefer_ceil"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0,
                        a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(2 * 1 * 4 * 8);
    // clang-format off
    std::vector<float> golden = { 
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        5.1f,  5.1f,    6.1f,  7.1f,    7.1f,  8.1f,    9.1f,  9.1f,  
        10.1f,  10.1f,  11.1f,  12.1f,  12.1f,  13.1f,  14.1f,  14.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        20.1f,  20.1f,  21.1f,  22.1f,  22.1f,  23.1f,  24.1f,  24.1f,  
        25.1f,  25.1f,  26.1f,  27.1f,  27.1f,  28.1f,  29.1f,  29.1f};
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_test_3_1_input)
{
    // same inputs and outputs as test 1
    // batch size 1, 1 color channel, resize 3x3 to 5x8
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {1, 1, 5, 8}},
                                           {"nearest_mode", "floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f,
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f, 
        6.5f, 6.5f, 6.5f, 6.5f, 7.5f, 7.5f, 7.5f, 8.5
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_upsample_test_4_1_input)
{
    // batch size 2, 1 color channel, resize 3x5 by 1.6x
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(2 * 3 * 5);
    std::iota(data.begin(), data.end(), 0.1);
    // should upscale to 2x1x4x8
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3, 5}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {}},
                                           {"scales", {1.0, 1.0, 1.601, 1.601}},
                                           {"nearest_mode", "round_prefer_ceil"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(2 * 1 * 4 * 8);
    // clang-format off
    std::vector<float> golden = { 
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        0.1f,  0.1f,    1.1f,  2.1f,    2.1f,  3.1f,    4.1f,  4.1f,  
        5.1f,  5.1f,    6.1f,  7.1f,    7.1f,  8.1f,    9.1f,  9.1f,  
        10.1f,  10.1f,  11.1f,  12.1f,  12.1f,  13.1f,  14.1f,  14.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        15.1f,  15.1f,  16.1f,  17.1f,  17.1f,  18.1f,  19.1f,  19.1f,  
        20.1f,  20.1f,  21.1f,  22.1f,  22.1f,  23.1f,  24.1f,  24.1f,  
        25.1f,  25.1f,  26.1f,  27.1f,  27.1f,  28.1f,  29.1f,  29.1f};
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_optimize_test)
{
    // matcher/optimized code should produce the same result as Resize op.
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    // non-matching sizes/scales attributes are ignored for 2 input arguments
    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {1}},
                                           {"scales", {1}},
                                           {"nearest_mode", "floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0,
                        a1);
    auto p2 = p;
    migraphx::run_passes(p,
                         {migraphx::split_single_dyn_dim{},
                          migraphx::simplify_dyn_ops{},
                          migraphx::dead_code_elimination{}});
    EXPECT(p != p2);
    auto result = p.eval({}).back();
    p.compile(migraphx::make_target("ref"));

    std::vector<float> res_data(1 * 1 * 5 * 8);
    // clang-format off
    std::vector<float> golden = {
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        0.5f, 0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f,
        3.5f, 3.5f, 3.5f, 3.5f, 4.5f, 4.5f, 4.5f, 5.5f, 
        6.5f, 6.5f, 6.5f, 6.5f, 7.5f, 7.5f, 7.5f, 8.5f
        };
    // clang-format on
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}

TEST_CASE(resize_fail_test_1)
{
    // invalid resize mode
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {0.75, 0.25, 1., 1.}},
                                               {"mode", "invalid"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            {a0});
    }));
}

TEST_CASE(resize_fail_test_2)
{
    // "sizes" attribute wrong vector size
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"sizes", {1, 2}},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            {a0});
    }));
}

TEST_CASE(resize_fail_test_3)
{
    // "scales" attribute wrong vector size
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1., 2.}},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            {a0});
    }));
}

TEST_CASE(resize_fail_test_4)
{
    // invalid coordinate_transformation_mode
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::module m;

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"scales", {1., 1., 0.75, 0.25}},
                                           {"nearest_mode", "round_prefer_floor"},
                                           {"coordinate_transformation_mode", "invalid"}}),
                        {a0});
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws([&] { p.eval({}); }));
}

TEST_CASE(resize_fail_test_5)
{
    // invalid nearest_mode
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};

    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::module m;

    mm->add_instruction(
        migraphx::make_op("resize",
                          {{"scales", {1., 1., 0.75, 0.25}},
                           {"nearest_mode", "invalid"},
                           {"coordinate_transformation_mode", "pytorch_half_pixel"}}),
        {a0});
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws([&] { p.eval({}); }));
}

TEST_CASE(resize_fail_test_6)
{
    // wrong dimension for second input
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {3}};
    std::vector<int> size_values = {1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            a0,
                            a1);
    }));
}

TEST_CASE(resize_fail_test_7)
{
    // wrong rank for second input
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4, 1}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            a0,
                            a1);
    }));
}

TEST_CASE(resize_fail_test_8)
{
    // dynamic second input
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape::dynamic_dimension dd{4, 4};
    migraphx::shape size_input{migraphx::shape::int32_type, {dd}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_parameter("Y", size_input);

    // a0 = input data
    // a1 = sizes of output
    EXPECT(test::throws([&] {
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            a0,
                            a1);
    }));
}
