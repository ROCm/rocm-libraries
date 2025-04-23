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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(batch_norm_3d_test)
{
    migraphx::program p = read_onnx("batch_norm_3d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {2, 2, 2, 2, 2}};
    migraphx::shape c_shape(migraphx::shape::half_type, {2});

    // using migraphx::half copy conversion since it doesn't have initializer_list constructor
    std::vector<float> tmp = {5., 5., 8., 7., 3., 4., 1., 7., 5., 5., 9., 4., 7., 2., 2., 2.,
                              6., 1., 4., 9., 2., 8., 0., 2., 1., 4., 8., 8., 3., 3., 0., 8.};
    std::vector<migraphx::half> x_data{tmp.cbegin(), tmp.cend()};
    tmp = {1., 1.};
    std::vector<migraphx::half> scale_data{tmp.cbegin(), tmp.cend()};
    tmp = {
        0.,
        0.,
    };
    std::vector<migraphx::half> bias_data{tmp.cbegin(), tmp.cend()};
    tmp = {-0.75, 0.29};
    std::vector<migraphx::half> mean_data{tmp.cbegin(), tmp.cend()};
    tmp = {0.31, 0.37};
    std::vector<migraphx::half> variance_data{tmp.cbegin(), tmp.cend()};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {10.33, 10.33, 15.71, 13.914, 6.734, 8.53,   3.143, 13.914, 7.742,   7.742, 14.32,
           6.098, 11.03, 2.81,  2.81,   2.81,  12.125, 3.143, 8.53,   17.52,   4.938, 15.71,
           1.347, 4.938, 1.167, 6.098,  12.67, 12.67,  4.453, 4.453,  -0.4768, 12.67};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
