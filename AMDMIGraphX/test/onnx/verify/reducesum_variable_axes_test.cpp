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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

static auto reducesum_variable_axes_test_base(const std::string& file, size_t axes_size)
{
    std::pair<std::vector<float>, migraphx::shape> ret;

    migraphx::onnx_options options;
    options.map_input_dims["axes"] = std::vector<size_t>{axes_size};
    migraphx::program p            = read_onnx(file, options);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    migraphx::shape x_shape{migraphx::shape::float_type, {3, 4, 5, 6}};
    std::vector<float> x(x_shape.elements());
    std::iota(x.begin(), x.end(), 0);
    pm["x"]        = migraphx::argument(x_shape, x.data());
    auto axes_data = axes_size == 0 ? std::vector<int64_t>{} : std::vector<int64_t>{2};
    pm["axes"]     = migraphx::argument(migraphx::shape{migraphx::shape::int64_type, {axes_size}},
                                    axes_data.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { ret.first.assign(output.begin(), output.end()); });
    ret.second = result.get_shape();

    return ret;
}

TEST_CASE(bla)
{
    auto [result_vector, shape] =
        reducesum_variable_axes_test_base("reducesum_variable_axes_test.onnx", 1);
    std::vector<float> gold{60,   65,   70,   75,   80,   85,   210,  215,  220,  225,  230,  235,
                            360,  365,  370,  375,  380,  385,  510,  515,  520,  525,  530,  535,
                            660,  665,  670,  675,  680,  685,  810,  815,  820,  825,  830,  835,
                            960,  965,  970,  975,  980,  985,  1110, 1115, 1120, 1125, 1130, 1135,
                            1260, 1265, 1270, 1275, 1280, 1285, 1410, 1415, 1420, 1425, 1430, 1435,
                            1560, 1565, 1570, 1575, 1580, 1585, 1710, 1715, 1720, 1725, 1730, 1735};

    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {3, 4, 1, 6}});
    EXPECT(result_vector == gold);
}

TEST_CASE(bla2)
{
    auto [result_vector, shape] =
        reducesum_variable_axes_test_base("reducesum_variable_axes_test.onnx", 0);
    std::vector<float> gold{64620};

    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}});
    EXPECT(result_vector == gold);
}

TEST_CASE(bla3)
{
    auto [result_vector, shape] =
        reducesum_variable_axes_test_base("reducesum_variable_axes_noop_test.onnx", 1);
    std::vector<float> gold{60,   65,   70,   75,   80,   85,   210,  215,  220,  225,  230,  235,
                            360,  365,  370,  375,  380,  385,  510,  515,  520,  525,  530,  535,
                            660,  665,  670,  675,  680,  685,  810,  815,  820,  825,  830,  835,
                            960,  965,  970,  975,  980,  985,  1110, 1115, 1120, 1125, 1130, 1135,
                            1260, 1265, 1270, 1275, 1280, 1285, 1410, 1415, 1420, 1425, 1430, 1435,
                            1560, 1565, 1570, 1575, 1580, 1585, 1710, 1715, 1720, 1725, 1730, 1735};

    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {3, 4, 1, 6}});
    EXPECT(result_vector == gold);
}

TEST_CASE(bla4)
{
    auto [result_vector, shape] =
        reducesum_variable_axes_test_base("reducesum_variable_axes_noop_test.onnx", 0);
    std::vector<float> gold(3 * 4 * 5 * 6);
    std::iota(gold.begin(), gold.end(), 0);

    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    EXPECT(result_vector == gold);
}
