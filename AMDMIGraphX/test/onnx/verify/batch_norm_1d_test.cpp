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

TEST_CASE(batch_norm_1d_test)
{
    migraphx::program p = read_onnx("batch_norm_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {2, 3, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {3});
    std::vector<float> tmp = {1.652,     -0.5103, 0.3254,  2.441,   2.084,    0.4497,
                              1.005,     -0.2401, -0.4307, 0.07623, -0.02927, 0.4236,
                              -0.004498, -0.4282, -0.5527, 0.02205, -1.472,   -1.7295,
                              0.796,     0.9507,  0.2312,  0.664,   -0.06964, 1.035};
    std::vector<migraphx::half> x_data{tmp.cbegin(), tmp.cend()};
    std::vector<float> scale_data    = {-1.336926, -1.0679098, 0.10368501};
    std::vector<float> bias_data     = {0.20240043, -0.70175606, -0.8859727};
    std::vector<float> mean_data     = {0.30854642, -0.36574763, -0.9463552};
    std::vector<float> variance_data = {0.43428132, 0.97773486, 0.30332062};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-2.523, 1.863,   0.1681,  -4.125, -3.348, -1.582, -2.182,  -0.8374,
           -0.789, -0.6934, -0.7134, -0.628, 0.8374, 1.697,  1.949,   0.7837,
           0.4927, 0.771,   -1.956,  -2.123, -0.664, -0.583, -0.7207, -0.5127};

    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
