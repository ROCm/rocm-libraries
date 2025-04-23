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

static auto reducesum_variable_dynamic_axes_test_base(migraphx::shape axes_shape,
                                                      std::vector<int64_t> axes_data,
                                                      const std::string& file)
{
    std::pair<std::vector<float>, migraphx::shape> ret;

    migraphx::onnx_options options;
    const std::vector<migraphx::shape::dynamic_dimension> axes_dims{{0, 3}};
    options.map_dyn_input_dims["axes"] = axes_dims;
    migraphx::program p                = read_onnx(file, options);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> x(x_shape.elements());
    std::iota(x.begin(), x.end(), 0);
    pm["x"] = migraphx::argument(x_shape, x.data());

    std::vector<int64_t> axes{1};
    pm["axes"] = migraphx::argument(axes_shape, axes_data.data());

    auto result = p.eval(pm).back();
    ret.second  = result.get_shape();
    result.visit([&](auto output) { ret.first.assign(output.begin(), output.end()); });
    return ret;
}

TEST_CASE(reducesum_variable_dynamic_axes_test)
{
    auto [result, shape] = reducesum_variable_dynamic_axes_test_base(
        {migraphx::shape::int64_type, {1}},
        std::vector<int64_t>{1},
        "reducesum_variable_dynamic_axes_verify_test.onnx");
    std::vector<float> gold{2, 4, 10, 12};
    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {2, 1, 2}});
    EXPECT(result == gold);
}

TEST_CASE(reducesum_variable_dynamic_axes_empty_test)
{
    auto [result, shape] = reducesum_variable_dynamic_axes_test_base(
        {migraphx::shape::int64_type, {0}},
        std::vector<int64_t>{},
        "reducesum_variable_dynamic_axes_verify_test.onnx");
    std::vector<float> gold{28};
    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {1, 1, 1}});
    EXPECT(result == gold);
}

TEST_CASE(reducesum_variable_dynamic_axes_noop_set_test)
{
    auto [result, shape] = reducesum_variable_dynamic_axes_test_base(
        {migraphx::shape::int64_type, {1}},
        std::vector<int64_t>{1},
        "reducesum_variable_dynamic_axes_noop_set_verify_test.onnx");
    std::vector<float> gold{2, 4, 10, 12};
    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {2, 1, 2}});
    EXPECT(result == gold);
}

TEST_CASE(reducesum_variable_dynamic_axes_empty_noop_set_test)
{
    auto [result, shape] = reducesum_variable_dynamic_axes_test_base(
        {migraphx::shape::int64_type, {0}},
        std::vector<int64_t>{},
        "reducesum_variable_dynamic_axes_noop_set_verify_test.onnx");
    std::vector<float> gold(8);
    std::iota(gold.begin(), gold.end(), 0);
    EXPECT(shape == migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    EXPECT(result == gold);
}
