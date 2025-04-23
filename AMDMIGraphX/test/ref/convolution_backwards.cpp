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

TEST_CASE(convolution_backwards_1d)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3}};
    std::vector<float> x_data{0, 0.5, 1};
    std::vector<float> w_data{0.5, 0.5, 0.5};

    std::vector<float> gold{0, 0.25, 0.75, 0.75, 0.5};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(migraphx::make_op("convolution_backwards",
                                          {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                        x,
                        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2d)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{0,  1,  3, 3,  2,  3,  8,  15, 12, 7,  9,  21, 36,
                            27, 15, 9, 20, 33, 24, 13, 6,  13, 21, 15, 8};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(migraphx::make_op("convolution_backwards"), x, w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

// test values from pytorch
TEST_CASE(convolution_backwards_3d)
{
    migraphx::shape s_1{migraphx::shape::float_type, {1, 1, 1, 2, 3}};
    migraphx::shape s_2{migraphx::shape::float_type, {1, 1, 3, 2, 3}};

    // clang-format off
    std::vector<float> x_data{0.8471, -0.4195, -2.2749, 1.2491, 0.1722, 0.3246};
    std::vector<float> w_data{
        0.6478, -0.1985, 0.0633, -0.3479, 2.7056, -0.1440,
        -1.1229, -0.7507, -1.3151, 0.8884, -0.1859, -0.3407,
        -1.1544, -1.5893, 1.6265, -1.4624, 0.3812, -1.5378
    };
    std::vector<float> gold{0.5488,  -0.4399, -1.3369, 0.4251,  -0.1439, 0.5145,  2.3015,  -0.2104,
                            -6.1482, 0.3482,  -0.4346, 3.3197,  0.1731,  0.8533,  -0.0467, -0.9512,
                            -0.1649, 1.7553,  2.2594,  2.9917,  -0.6500, -1.6612, -4.3680, 0.0957,
                            0.3482,  1.1097,  -0.0792, -0.1692, -0.1190, -0.1106, -0.9779, -0.8621,
                            4.6707,  2.9332,  -3.7001, -2.6808, -1.2476, 3.2475,  -0.4578, 4.0263,
                            -1.8267, 0.2243,  -2.3299, -0.1411, -0.4991};
    // clang-format on

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s_1, x_data});
    auto w   = mm->add_literal(migraphx::literal{s_2, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_padding1)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{8, 15, 12, 21, 36, 27, 20, 33, 24};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_padding2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<float> gold{3., 8., 15., 12., 7., 9., 21., 36., 27., 15., 9., 20., 33., 24., 13.};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2stride)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> gold{0.,  0., 1., 1., 3.,  2.,  2.,  0.,  0.,  1., 1., 3.,  2.,
                            2.,  3., 3., 8., 5.,  12., 7.,  7.,  3.,  3., 7., 4.,  9.,
                            5.,  5., 9., 9., 20., 11., 24., 13., 13., 6., 6., 13., 7.,
                            15., 8., 8., 6., 6.,  13., 7.,  15., 8.,  8.};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_2dilation)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> gold{0., 1., 2., 1.,  2.,  1.,  2.,  3.,  4.,  8., 4., 8., 4.,
                            5., 6., 8., 16., 8.,  16., 8.,  10., 3.,  4., 8., 4., 8.,
                            4., 5., 6., 8.,  16., 8.,  16., 8.,  10., 3., 4., 8., 4.,
                            8., 4., 5., 6.,  7.,  14., 7.,  14., 7.,  8.};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{s, x_data});
    auto w   = mm->add_literal(migraphx::literal{s, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {2, 2}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

// test values from pytorch
TEST_CASE(convolution_backwards_2d_group)
{
    migraphx::shape activations_shape{migraphx::shape::float_type, {1, 2, 3, 3}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {2, 2, 3, 3}};
    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<float> w_data(36, 1.0);

    std::vector<float> gold{
        0.,  1.,  3.,   3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21.,  36., 27., 15., 9.,  20.,
        33., 24., 13.,  6.,  13., 21., 15., 8.,  0.,  1.,  3.,  3.,   2.,  3.,  8.,  15., 12.,
        7.,  9.,  21.,  36., 27., 15., 9.,  20., 33., 24., 13., 6.,   13., 21., 15., 8.,  9.,
        19., 30., 21.,  11., 21., 44., 69., 48., 25., 36., 75., 117., 81., 42., 27., 56., 87.,
        60., 31., 15.,  31., 48., 33., 17., 9.,  19., 30., 21., 11.,  21., 44., 69., 48., 25.,
        36., 75., 117., 81., 42., 27., 56., 87., 60., 31., 15., 31.,  48., 33., 17.};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_literal(migraphx::literal{activations_shape, x_data});
    auto w   = mm->add_literal(migraphx::literal{weights_shape, w_data});

    mm->add_instruction(migraphx::make_op("convolution_backwards", {{"group", 2}}), x, w);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_dyn_batch1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // clang-format off
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {1, 1}, {3, 3}, {3, 3}}};
    // clang-format on
    auto x = mm->add_parameter("x", s);
    auto w = mm->add_parameter("w", s);

    mm->add_instruction(migraphx::make_op("convolution_backwards"), x, w);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> w_data{1, 1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    params["w"] = migraphx::argument(input_fixed_shape, w_data.data());
    auto result = p.eval(params).back();

    std::vector<float> gold{0,  1,  3, 3,  2,  3,  8,  15, 12, 7,  9,  21, 36,
                            27, 15, 9, 20, 33, 24, 13, 6,  13, 21, 15, 8};
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(convolution_backwards_dyn_batch2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // clang-format off
    migraphx::shape x_shape{migraphx::shape::float_type,
                      {{1, 4}, {1, 1}, {5, 5}, {5, 5}}};
    // clang-format on
    auto x = mm->add_parameter("x", x_shape);
    migraphx::shape w_shape{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> w_data(9, 1.);
    auto w = mm->add_literal(migraphx::literal{w_shape, w_data});

    mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {2, 2}}, {"stride", {2, 2}}, {"dilation", {2, 2}}}),
        x,
        w);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x_data(25);
    std::iota(x_data.begin(), x_data.end(), 0.);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {1, 1, 5, 5}};
    params["x"] = migraphx::argument(input_fixed_shape, x_data.data());
    auto result = p.eval(params).back();

    //clang-format off
    std::vector<float> gold{12.,  0., 21.,  0., 27.,  0., 33.,  0., 24.,  0., 0.,  0., 0.,   0.,
                            0.,   0., 0.,   0., 33.,  0., 54.,  0., 63.,  0., 72., 0., 51.,  0.,
                            0.,   0., 0.,   0., 0.,   0., 0.,   0., 63.,  0., 99., 0., 108., 0.,
                            117., 0., 81.,  0., 0.,   0., 0.,   0., 0.,   0., 0.,  0., 93.,  0.,
                            144., 0., 153., 0., 162., 0., 111., 0., 0.,   0., 0.,  0., 0.,   0.,
                            0.,   0., 72.,  0., 111., 0., 117., 0., 123., 0., 84.};
    //clang-format on

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
