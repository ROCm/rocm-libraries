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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(avgpool_rank3_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {1};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.3, 0.25, 0.65, 0.7, 0.5, 0.4, 0.4, 0.35};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_dil_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {2};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.35, 0.15, 0.85, 0.3, 0.1, 0.65};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_dil_test2)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.2, 0.45, 0.35};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_pad_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {1};
    op.stride    = {1};
    op.dilations = {1};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        0.3, 0.25, 0.3, 0.25, 0.1, 0.8, 0.65, 0.7, 0.5, 0.1, 0.1, 0.4, 0.4, 0.35, 0.6};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_pad_dil_test)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {1};
    op.stride    = {1};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.2, 0.2, 0.9, 0.45, 0.5, 0.1, 0.35, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_test)
{
    // Dynamic input, no padding
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}},
                                           {"dilations", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.3, 0.25, 0.65, 0.7, 0.5, 0.4, 0.4, 0.35};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_pad_test)
{
    // Dynamic input with explicit padding
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 3}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2}},
                                           {"padding", {1}},
                                           {"stride", {1}},
                                           {"dilations", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{
        0.3, 0.25, 0.3, 0.25, 0.1, 0.8, 0.65, 0.7, 0.5, 0.1, 0.1, 0.4, 0.4, 0.35, 0.6};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_auto_pad_test)
{
    // Pooling with dynamic input, multidimensional kernel and auto-padding
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s =
        migraphx::shape{migraphx::shape::float_type, {{1, 1}, {1, 3}, {2, 6, {2}}, {2, 6, {2}}}};
    auto x = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {
                              {"mode", migraphx::op::pooling_mode::average},
                              {"dyn_global", false},
                              // non-default auto padding
                              {"padding_mode", migraphx::op::padding_mode_t::same_upper},
                              {"lengths", {2, 3}},
                          }),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4};

    //      * 1 2 *      auto padding should look like this
    //      * 3 4 *
    //      * * * *

    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.5, 2.5, 3.5, 3.5};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_auto_pad_1d_test)
{
    // Dynamic input with auto padding (== padding_mode specified)
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 3}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::average},
                           {"lengths", {2}},
                           //    padding added will be {1, 0} to make output
                           //    the same size as input
                           {"padding_mode", migraphx::op::padding_mode_t::same_lower},
                           {"stride", {1}},
                           {"dilations", {1}}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold{0.3, 0.25, 0.3, 0.25, 
                            0.8, 0.65, 0.7, 0.5, 
                            0.1, 0.4,  0.4, 0.35};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_dyn_pad_ceil_test)
{
    // pooling with dynamic input and padding
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 3}, {2, 4}, {2, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"lengths", {2, 3}},
                                           {"padding", {1, 2}},
                                           {"ceil_mode", true},
                                           {"stride", {1, 1}},
                                           {"dilations", {1, 1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4};

    //  * *  *  * * *
    //  * *  1  2 * *      padded input will look like this
    //  * *  3  4 * *      but the * are ignored in averaging
    //  * *  *  * * *

    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // clang-format off
    std::vector<float> gold{1.0, 1.5, 1.5, 2.0, 
                            2.0, 2.5, 2.5, 3.0, 
                            3.0, 3.5, 3.5, 4.0};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank3_stride2_test)
{
    // 1D case 2, stride 2
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2};
    op.padding   = {1};
    op.stride    = {2};
    op.dilations = {1};

    // clang-format off
    std::vector<float> data{1.6321, -2.4186, 0.2239, -1.4232, 
                            0.8158, 0.4103, -0.3149, -0.1361,
                            -0.3442, 2.007, 0.4331, 1.5295,
                            0.9965, 0.4766, 1.0942, -0.2915};
    // clang-format on
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold{1.6321, -1.09735, -1.4232,
                            0.8158, 0.0477, -0.1361, 
                            -0.3442, 1.22005, 1.5295,
                            0.9965, 0.7854, -0.2915};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(avgpool_rank5_test)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    op.lengths   = {2, 2, 2};
    op.padding   = {0, 0, 0};
    op.stride    = {1, 1, 1};
    op.dilations = {1, 1, 1};

    std::vector<float> data{
        -0.179, -1.756, 0.651,  1.955,  1.87,   -0.604, 0.247,  0.449,  -0.137, 1.187,  1.593,
        0.424,  2.698,  -0.104, -0.069, -1.293, 0.538,  1.291,  0.974,  1.096,  0.74,   -0.669,
        -1.08,  -1.041, -1.407, 1.43,   -0.211, -0.017, 0.532,  1.276,  0.627,  0.236,  -0.396,
        -0.204, 0.501,  -0.599, -1.414, -0.615, -0.274, 0.168,  -0.144, 0.5,    1.42,   1.082,
        -0.952, -0.846, -1.244, 1.475,  1.246,  1.344,  -1.722, -1.24,  -0.851, 0.06,   0.507,
        0.762,  -0.007, -1.484, 1.028,  0.317,  1.077,  -1.289, 0.875,  -0.417, -0.673, 1.715,
        -0.307, 0.264,  -0.973, 1.412,  2.561,  -0.515, -0.201, 0.827,  -1.231, 1.958,  -0.552,
        0.036,  -0.993, -0.859, -1.458, -0.575, 0.048,  -0.779, -1.025, -1.135, 1.166,  -0.131,
        0.726,  0.52,   0.467,  -0.494, 0.675,  0.203,  -0.63,  -0.918, -0.5,   -1.395, 1.39,
        1.705,  0.444,  -0.835, -0.506, 0.101,  0.602,  0.543,  0.357,  1.042};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{
        0.908,     0.250625,  0.795,     0.40425, 0.711875,  0.194875,  0.014125,  0.09425,
        -0.078375, 0.139375,  0.46075,   0.0285,  -0.188125, -0.085,    0.378125,  -0.085375,
        -0.04,     0.304125,  0.40775,   0.2835,  0.112375,  -0.073375, 0.4355,    -0.187,
        -0.392625, -0.258375, -0.485875, -0.0345, 0.16125,   -0.131875, -0.228375, 0.068625};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globalavgpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::average};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globalavgpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6}, {2, 6, {2}}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::average}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.25, 0.575, 0.375};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globallppool_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op     = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    auto lens   = s.lens();
    op.lengths  = {lens[2], lens[3]};
    op.lp_order = 2;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5477225575051662, 1.307669683062202, 0.9327379053088815};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globallppool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s =
        migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6, {2}}, {2, 6, {2}}}};
    auto x = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::lpnorm}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5477225575051662, 1.307669683062202, 0.9327379053088815};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    auto lens  = s.lens();
    op.lengths = {lens[2], lens[3]};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(globalmaxpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s =
        migraphx::shape{migraphx::shape::float_type, {{1, 1}, {3, 3}, {2, 6, {2}}, {2, 6, {2}}}};
    auto x = mm->add_parameter("X", s);
    mm->add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"dyn_global", true}}),
        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 2, 2}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.9, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(lppool_l1_norm_test)
{
    // L1 norm test
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {1};
    op.lp_order  = 1;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.5, 0.6, 0.5, 1.3, 1.4, 1.0, 0.8, 0.8, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

// TODO: this tests compliance with a oneDNN rule and a feature that's commented out
// in pooling.hpp
// TEST_CASE(lppool_l1_norm_err_test)
// {
//     // padding too large for kernel size
//     migraphx::program p;
//     auto* mm     = p.get_main_module();
//     auto s       = migraphx::shape{migraphx::shape::float_type, {1, 2, 5}};
//     auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
//     op.lengths   = {3};
//     op.padding   = {2};
//     op.stride    = {1};
//     op.dilations = {1};
//     op.lp_order  = 1;

//     std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7};
//     auto l0 = mm->add_literal(migraphx::literal{s, data});
//     EXPECT(test::throws([&] {
//             mm->add_instruction(op, l0);
//         }));
// }

TEST_CASE(lppool_l2_norm_test)
{
    // L2 norm test
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {1};
    op.lp_order  = 2;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.36055512754639896,
                            0.447213595499958,
                            0.4123105625617661,
                            0.9433981132056605,
                            1.0295630140987,
                            0.9055385138137417,
                            0.7071067811865475,
                            0.7071067811865475,
                            0.6082762530298219};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(lppool_l2_norm_test_asym_padding)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    op.lengths   = {2, 1};
    op.padding   = {1, 0, 0, 0};
    op.stride    = {1, 1};
    op.dilations = {1, 1};
    op.lp_order  = 2;

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9,  0.1, 0.1, 0.7,
                            0.1, 0.6, 0.7, 0.5, 0.1, 0.2, 0.11, 0.2, 0.3, 0.4};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.3,        0.2,        0.4,        0.1,        0.8544004,
                            0.53851646, 0.98488575, 0.14142136, 0.8062258,  0.86023253,
                            0.9055385,  0.60827625, 0.70710677, 0.86023253, 0.14142136,
                            0.6324555,  0.70859015, 0.53851646, 0.31622776, 0.44721362};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(lppool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::lpnorm},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}},
                                           {"dilations", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.36055512754639896,
                            0.447213595499958,
                            0.4123105625617661,
                            0.9433981132056605,
                            1.0295630140987,
                            0.9055385138137417,
                            0.7071067811865475,
                            0.7071067811865475,
                            0.6082762530298219};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        -2.1314404,  -1.63041711, 1.54562736,  1.04625261,  -1.42931843, -0.48703974, 0.4065806,
        -0.1524526,  1.30775225,  0.45538983,  -0.06631992, -1.75332725, 1.33493888,  0.47327688,
        0.36873096,  1.18358743,  -0.34640595, 1.22098756,  0.01946825,  -0.20238149, 0.43348005,
        -0.67991608, -0.83041084, 0.93537551,  0.70241445,  -0.5654031,  -1.30899191, -0.26735824,
        -0.52444768, 1.99097753,  1.86504853,  -0.26506025, 0.26236168,  0.43763575,  0.95300823,
        -1.02733946, -0.74655169, -0.5374338,  -0.28901565, -0.59789604, 0.5310151,   0.99125904,
        0.40609556,  -1.57175648, 0.22031412,  1.45862222,  0.53217483,  1.39087725,  1.00170159,
        -0.87175864, -1.7204628,  -1.72008383, -0.38656762, -0.01443311, 1.46645272,  -1.39995027,
        0.22505587,  -0.43461126, -0.05511411, -0.79950953, -0.01439556, 0.08795211,  1.18943918,
        -0.84079367, -1.73383629, -0.55662078, -0.30626822, -0.67339015, 0.44179603,  0.54316711,
        0.40899998,  -0.27831686, -1.11900508, -0.0881724,  0.35483059,  2.36277103,  -0.04765317,
        -0.36865309, 0.73814237,  1.47151589,  1.36546791,  -0.32649881, -1.0517807,  2.24768877,
        0.68883753,  0.58646208,  -0.91017133, -0.50462508, -0.4013325,  -0.72348958, -0.47368807,
        0.35285577,  -1.01817429, -0.5152272,  0.60321307,  0.43521205,  -0.23733577, 0.66427642,
        0.82949388,  0.82443929,  0.71550399,  0.34561086,  0.68570769,  -0.40718508, -1.20350206,
        0.15793853,  -2.31013632, -0.07934658, -0.09348056, 0.36576006,  2.46601582,  0.11090943,
        0.9144392,   0.56759721,  -0.22112127, -0.21955389, 0.72474903,  -1.28448462, 1.53285873,
        0.37437943,  0.31409341,  1.95433736,  0.91620457,  0.86205518,  1.24365854,  0.19248386,
        0.22526583,  0.13462132,  -0.27561715, -2.06446075, -0.02306402, -1.38278747, 1.1411345,
        1.31293464,  -1.86041689, 1.06763375,  -0.26541466, 1.4545635,   1.11430049,  -0.66491818,
        0.87101674,  0.67768967,  -1.02062869, -1.05031872, -2.2764678,  -2.0200038,  0.37592548,
        -0.26701379, -0.83388507, 0.19403623,  1.00968623,  0.11020003,  1.16736257,  -1.1160326,
        0.47346735,  0.6126079,   -0.19135755, 1.33624589,  -0.29802522, -0.57873946, -1.06555879,
        -0.20686582, 1.36892557,  -0.19937795, 0.8649236,   -1.40126073, 1.53441942,  0.34682792,
        -1.31724346, -1.32898355, 2.40126371,  0.07845283,  1.35732043,  -0.63678312, 0.39429256,
        -1.36487007, -0.31026676, -0.44981545, -0.28994772, -0.14657612, -1.75206447, -0.70612341,
        1.20071781,  -1.64647579, -0.7133292,  0.88494766,  0.52119428,  -2.77387547, 2.07681108,
        -0.90133125, 0.2847338,   0.6174528,   -0.20616426, -0.64263535, -1.08496261, 0.54275119,
        -0.88503587, 0.6629802,   1.47319221,  -1.05829155, -0.97027361, -0.93187737, -1.39954746,
        -0.52359426, -0.14743951, 1.51522756,  0.2078452,   -1.28156149, -1.19363916, -0.78680223,
        -0.89094824, 1.30212069,  -0.77974445, -0.58411664, 0.48764706,  -0.67132682};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 6, 6}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {0, 0}},
                                           {"stride", {2, 2}},
                                           {"lengths", {3, 2}},
                                           {"dilations", {1, 1}}}),
                        al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(36);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        1.33493888, 1.54562736, 1.22098756, 1.33493888, 1.18358743, 1.99097753,
        1.00170159, 1.45862222, 1.39087725, 1.46645272, 1.18943918, -0.01443311,
        1.47151589, 2.36277103, 2.24768877, 0.68883753, 0.82949388, 0.71550399,
        1.95433736, 2.46601582, 1.53285873, 1.95433736, 1.06763375, 1.4545635,
        1.33624589, 1.16736257, 0.6126079,  1.36892557, 2.40126371, 1.53441942,
        0.52119428, 2.07681108, 0.88494766, 1.51522756, 0.54275119, 0.6629802};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_pad_test)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {-6, -5, -4, -3, -5, -1, 0, 1, 2, 3, 4, 5};
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 3, 2}};
    auto al = mm->add_literal(migraphx::literal{a_shape, a});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {1, 1}},
                                           {"stride", {2, 2}},
                                           {"lengths", {3, 2}},
                                           {"dilations", {1, 1}}}),
                        al);

    //   * *  *  *                                           * *  *  *
    //   * -6 -5 *                                           * 0  1  *
    //   * -4 -3 *      padding will look like this          * 2  3  *
    //   * -5 -1 *                  and this                 * 4  5  *
    //   * *  *  *      The * values are actually -INF       * *  *  *

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-4, -3, -4, -1, 2, 3, 4, 5};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_test0)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {1};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.3, 0.4, 0.4, 0.8, 0.9, 0.9, 0.7, 0.7, 0.6};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_test1)
{
    // 1D case 2, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {2};
    op.dilations = {1};

    std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, -0.6186, -0.9618,
                            0.6022, -0.1912, 1.1925,  0.5493,  0.1692,  -0.8039, -1.0281,
                            0.9907, 0.477,   1.5001,  -1.1603, -1.361,  1.2556};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4975, -0.0405, -0.6186, 0.6022, 0.5493, -0.8039, 1.5001, -1.1603};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_test2)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {1};
    op.dilations = {2};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.2, 0.9, 0.5, 0.1, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_test4)
{
    // 1D case 1, input is 3D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {1, 3, 4}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {1};
    op.stride    = {1};
    op.dilations = {3};

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.3, 0.2, 0.9, 0.8, 0.5, 0.1, 0.6, 0.7};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank3_ceil_test)
{
    // 1D case 2, input is 3D, ceil mode
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 5}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2};
    op.padding   = {0};
    op.stride    = {2};
    op.dilations = {1};
    op.ceil_mode = true;

    // clang-format off
    std::vector<float> data{0.4975, -0.1226, -0.0405, -0.2861, -0.1227, 
                        -0.6186, -0.9618, 0.6022, -0.1912, 1.1925,
                        0.5493,  0.1692,  -0.8039, -1.0281, 0.9907, 
                        0.477,   1.5001,  -1.1603, -1.361,  1.2556};
    // clang-format on
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    // clang-format off
    std::vector<float> gold{0.4975, -0.0405, -0.1227, -0.6186,
                            0.6022, 1.1925, 0.5493, -0.8039,
                            0.9907, 1.5001, -1.1603, 1.2556};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_rank5_test)
{
    // 3D, input is 5D
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto s       = migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 3, 3}};
    auto op      = migraphx::op::pooling{migraphx::op::pooling_mode::max};
    op.lengths   = {2, 2, 2};
    op.padding   = {0, 0, 0};
    op.stride    = {2, 2, 2};
    op.dilations = {1, 1, 1};

    std::vector<float> data{
        -2.8029, 0.5861,  0.7015,  0.1297,  -1.44,   -1.9472, 0.7812,  2.408,   -0.3145, 0.3405,
        -0.9146, 0.0624,  1.5064,  -0.8345, 1.7977,  1.8949,  1.0073,  -0.2102, -0.042,  -0.7146,
        0.6227,  -0.5263, -2.2598, 0.1713,  0.449,   0.5303,  -0.8622, -0.5691, 0.907,   -0.0569,
        -1.5348, -0.4109, -0.1461, -0.5445, 0.4266,  0.2282,  1.3655,  -2.1519, 0.6068,  -0.2001,
        -0.4702, 0.3864,  1.7083,  0.9096,  0.4286,  -1.8866, 0.7034,  0.0293,  1.4587,  0.7672,
        -2.8614, 0.8124,  -0.053,  1.0449,  0.845,   -0.0131, 0.1139,  -0.859,  -1.2681, -0.6337,
        -0.4644, 0.1938,  0.2889,  0.9035,  0.7118,  -0.5767, 0.4577,  -0.0549, 0.2237,  0.5756,
        0.0677,  -0.0223, -0.329,  0.2364,  2.7666,  -0.7417, -1.3196, -0.2655, 0.1698,  -0.1777,
        -0.9427, 2.6859,  -0.7501, 0.5175,  1.0029,  -2.6436, -0.4388, -1.2348, -0.1539, -0.6229,
        -0.4136, 0.5085,  0.4136,  -0.6439, -1.1953, -0.406,  -0.0195, 0.1869,  -0.8664, 1.1364,
        0.5041,  0.0647,  0.1941,  -1.0819, -0.4629, -0.5107, 0.3612,  -0.3583};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(op, l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1.5064, 1.3655, 0.9035, 2.6859};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}},
                                           {"dilations", {1}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.3, 0.4, 0.4, 0.8, 0.9, 0.9, 0.7, 0.7, 0.6};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(maxpool_dyn_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s   = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {4, 4}}};
    auto x   = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"lengths", {2}},
                                           {"padding", {0}},
                                           {"stride", {1}},
                                           {"dilations", {2}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0.3, 0.2, 0.4, 0.1, 0.8, 0.5, 0.9, 0.1, 0.1, 0.7, 0.1, 0.6};
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 3, 4}};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0.4, 0.2, 0.9, 0.5, 0.1, 0.7};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
