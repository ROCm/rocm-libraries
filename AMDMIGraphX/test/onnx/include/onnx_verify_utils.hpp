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

#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_VERIFY_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_VERIFY_UTILS_HPP

#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

template <typename T = float>
std::vector<T> norm_test(const std::vector<size_t>& x_dims,
                         std::vector<T>& scale,
                         std::vector<T>& bias,
                         migraphx::program p,
                         const std::string& scale_str = std::string{"scale"},
                         const std::string& bias_str  = std::string{"bias"})
{
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::get_type<T>{}, x_dims};
    migraphx::shape s_s{migraphx::shape::get_type<T>{}, {scale.size()}};
    migraphx::shape s_b{migraphx::shape::get_type<T>{}, {scale.size()}};

    std::vector<T> x(s_x.elements());
    std::iota(std::begin(x), std::end(x), 1);

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp[scale_str] = migraphx::argument(s_s, scale.data());
    pp[bias_str]  = migraphx::argument(s_b, bias.data());

    auto result = p.eval(pp).back();

    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

template <typename T = float>
std::vector<T> mvn_test(std::vector<size_t> data_lens, migraphx::program p)
{
    p.compile(migraphx::make_target("ref"));

    migraphx::shape data_shape(migraphx::shape::get_type<T>{}, std::move(data_lens));
    std::vector<T> data(data_shape.elements());
    std::iota(begin(data), end(data), 0);

    migraphx::parameter_map pm;
    pm["data"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pm).back();
    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

inline std::vector<float> gen_trilu_test(const migraphx::shape& s, const migraphx::program& p)
{
    // input data filled with values 1 to nelements
    std::vector<float> x_data(s.elements());
    std::iota(x_data.begin(), x_data.end(), 1);

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

#endif
