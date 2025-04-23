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

TEST_CASE(gemm_test)
{
    migraphx::program p = read_onnx("gemm_brcst_C_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a_shape{migraphx::shape::float_type, {5, 6}};
    std::vector<float> a_data = {0.26472837, 0.8525864,  0.41929847, 0.14151508, 0.43216065,
                                 0.67468566, 0.42488748, 0.82021785, 0.9782456,  0.5794279,
                                 0.6627283,  0.4790396,  0.9237051,  0.7340607,  0.67379653,
                                 0.87168175, 0.37324256, 0.33278653, 0.42736676, 0.024699844,
                                 0.75851107, 0.48719302, 0.5834426,  0.6938476,  0.43747696,
                                 0.24054702, 0.26912406, 0.6760658,  0.5419149,  0.89949054};

    migraphx::shape b_shape{migraphx::shape::float_type, {5, 7}};
    std::vector<float> b_data = {
        0.65727437,  0.54262096, 0.14126152, 0.8994123,  0.21831702,  0.81191784, 0.9371278,
        0.3438551,   0.7121373,  0.90316695, 0.26614252, 0.80144906,  0.80301756, 0.49930334,
        0.0719704,   0.63484156, 0.7343097,  0.32130218, 0.7094916,   0.6116475,  0.74144083,
        0.021210382, 0.38724765, 0.44830495, 0.62347615, 0.022489505, 0.23316588, 0.76540905,
        0.895689,    0.81540287, 0.223875,   0.9275573,  0.4621397,   0.70785195, 0.5658555};

    migraphx::shape c_shape{migraphx::shape::float_type, {6, 1}};
    std::vector<float> c_data = {
        0.07358502, 0.13792239, 0.8574055, 0.40553397, 0.38205826, 0.62062204};

    migraphx::parameter_map params;
    params["A"] = migraphx::argument(a_shape, a_data.data());
    params["B"] = migraphx::argument(b_shape, b_data.data());
    params["C"] = migraphx::argument(c_shape, c_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.45261115, 0.83629227, 0.7533463,  0.7189715, 0.69160205, 0.824082,  0.9187499,
        0.6659525,  0.96956736, 0.84293026, 0.8400868, 0.84835225, 1.0982862, 1.0642393,
        1.1447254,  1.6184721,  1.6048342,  1.4741788, 1.4334437,  1.638659,  1.7428316,
        0.8098607,  1.2157929,  1.1010075,  1.0706307, 1.0429881,  1.1771785, 1.2362702,
        0.8239243,  1.1112559,  0.9639262,  1.0813537, 0.8825792,  1.121141,  1.1885703,
        1.2227502,  1.4568202,  1.1388762,  1.55058,   1.0958102,  1.4637487, 1.5756242};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
