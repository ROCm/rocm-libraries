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

TEST_CASE(gemm_half_test)
{
    migraphx::program p = read_onnx("gemm_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a_shape{migraphx::shape::half_type, {8, 6}};
    std::vector<float> tmp = {0.2646, 0.8525, 0.4192, 0.1415, 0.4321,  0.675,  0.4248, 0.8203,
                              0.978,  0.5796, 0.6626, 0.479,  0.924,   0.734,  0.674,  0.8716,
                              0.3733, 0.3328, 0.4272, 0.0247, 0.7583,  0.4873, 0.5835, 0.694,
                              0.4375, 0.2406, 0.269,  0.6763, 0.542,   0.8994, 0.657,  0.5425,
                              0.1412, 0.8994, 0.2183, 0.812,  0.937,   0.3438, 0.712,  0.9033,
                              0.266,  0.8013, 0.803,  0.4993, 0.07196, 0.635,  0.7344, 0.3213};
    std::vector<migraphx::half> a_data{tmp.cbegin(), tmp.cend()};

    migraphx::shape b_shape{migraphx::shape::half_type, {8, 7}};
    tmp = {0.7095,  0.612,  0.741,  0.02121, 0.3872, 0.4482,  0.6235,  0.02249, 0.2332, 0.7656,
           0.8955,  0.8154, 0.2239, 0.9277,  0.4622, 0.708,   0.566,   0.0736,  0.138,  0.8574,
           0.4055,  0.382,  0.6206, 0.424,   0.3674, 0.435,   0.998,   0.3594,  0.701,  0.6216,
           0.01826, 0.6313, 0.514,  0.1095,  0.3203, 0.01636, 0.537,   0.01952, 0.4502, 0.8965,
           0.5415,  0.7456, 0.793,  0.756,   0.9,    0.5264,  0.05368, 0.4214,  0.276,  0.1517,
           0.08453, 0.83,   0.417,  0.1682,  0.845,  0.1729};
    std::vector<migraphx::half> b_data{tmp.cbegin(), tmp.cend()};

    migraphx::shape c_shape{migraphx::shape::half_type, {6, 1}};
    tmp = {0.10846, 0.672, 0.527, 0.94, 0.429, 0.2291};
    std::vector<migraphx::half> c_data{tmp.cbegin(), tmp.cend()};

    migraphx::parameter_map params;
    params["A"] = migraphx::argument(a_shape, a_data.data());
    params["B"] = migraphx::argument(b_shape, b_data.data());
    params["C"] = migraphx::argument(c_shape, c_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {1.071, 1.378, 1.465, 1.093, 0.968, 1.542, 1.145, 1.287,  1.533, 1.75,  1.338,
           1.449, 1.592, 1.668, 1.265, 1.531, 1.656, 1.348, 1.2705, 1.525, 1.479, 1.754,
           2.143, 2.062, 1.921, 1.836, 2.203, 1.952, 1.055, 1.225,  1.418, 1.209, 1.155,
           1.42,  1.234, 1.302, 1.593, 1.368, 1.289, 1.327, 1.451,  1.394};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
