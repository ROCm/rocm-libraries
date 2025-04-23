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

TEST_CASE(batch_norm_2d_test)
{
    migraphx::program p = read_onnx("batch_norm_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {3});
    std::vector<float> x_data = {
        1.6524342,   -0.51048076, 0.32543048,  2.4410043,   2.0833702,   0.44981122,  1.0044622,
        -0.24006313, -0.43065986, 0.07626268,  -0.02927135, 0.42347777,  -0.00449735, -0.4281568,
        -0.5527635,  0.02204161,  -1.4719028,  -1.7298799,  0.79596406,  0.9505461,   0.23115851,
        0.6639593,   -0.06963254, 1.0348768,   -1.336926,   -1.0679098,  0.10368501,  0.20240043,
        -0.70175606, -0.8859727,  0.30854642,  -0.36574763, -0.9463552,  0.9476916,   0.37686515,
        -0.05184272, -0.7151244,  -0.37341377, 0.59440356,  0.10051094,  -0.20755945, 0.9098465,
        1.1664004,   1.4075205,   -1.1522529,  -0.34607422, 0.32027543,  -0.6885485,  0.5404544,
        0.10012514,  0.8767704,   1.0032021,   -1.2755303,  0.23577735,  0.74239916,  1.0146079,
        0.60875916,  -0.29163074, 1.4872868,   0.20466477,  -0.26367408, -0.56394804, -0.56043875,
        0.7763664,   -0.9626441,  0.29653943,  -3.2231965,  0.03322164,  0.03402911,  0.77308357,
        -0.0654009,  -0.30463725, 0.22182712,  -0.22594836, -0.5807543,  -0.22390617, -0.24484141,
        -2.0761833,  1.8459716,   0.2455878,   0.99913245,  -0.9266217,  -0.1938893,  0.6417983,
        -1.0880078,  0.49565446,  2.1584804,   1.2276239,   3.3091128,   0.14217089,  0.9425477,
        0.07578196,  0.4067431,   0.71984154,  -0.20796849, 0.90003085};

    std::vector<float> scale_data = {0.658487, 0.03700604, 2.463201};

    std::vector<float> bias_data = {0.03497279, 0.17080553, 0.5636415};

    std::vector<float> mean_data = {0.1954783, 0.6203974, 0.8116831};

    std::vector<float> variance_data = {0.30558077, 0.04536599, 0.05461315};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1.77046824e+00,  -8.05950999e-01, 1.89769119e-01,  2.70979643e+00,  2.28379035e+00,
        3.37928861e-01,  9.98617530e-01,  -4.83835101e-01, -7.10869908e-01, -1.07034385e-01,
        -2.32744321e-01, 3.06560963e-01,  -2.03234047e-01, -7.07888365e-01, -8.56317282e-01,
        -1.71621382e-01, -1.92677066e-01, -2.37493858e-01, 2.01305658e-01,  2.28160262e-01,
        1.03185430e-01,  1.78373277e-01,  5.09308279e-02,  2.42810518e-01,  -1.69228360e-01,
        -1.22493818e-01, 8.10402334e-02,  9.81894583e-02,  -5.88841513e-02, -9.08869803e-02,
        1.16629556e-01,  -5.11445105e-04, -1.79648399e+01, 1.99707508e+00,  -4.01903248e+00,
        -8.53731060e+00, -1.55278311e+01, -1.19264421e+01, -1.72633123e+00, -6.93161058e+00,
        -1.01784554e+01, 1.59821415e+00,  4.30211163e+00,  6.84334660e+00,  -2.01348572e+01,
        -1.16383028e+01, -4.61544800e+00, -1.52477398e+01, 4.45901126e-01,  -7.86099210e-02,
        8.46513629e-01,  9.97116446e-01,  -1.71726203e+00, 8.29761624e-02,  6.86453462e-01,
        1.01070285e+00,  5.27264357e-01,  -5.45261383e-01, 1.57374811e+00,  4.59154993e-02,
        -5.11959970e-01, -8.69639993e-01, -8.65459919e-01, 7.26914644e-01,  -1.04206637e-01,
        1.14543661e-01,  -4.96918678e-01, 6.87990561e-02,  6.89393356e-02,  1.97330773e-01,
        5.16659655e-02,  1.01048872e-02,  1.01564340e-01,  2.37750299e-02,  -3.78632471e-02,
        2.41298079e-02,  2.04928555e-02,  -2.97655046e-01, 3.83717060e-01,  1.05692141e-01,
        2.53922558e+00,  -1.77568626e+01, -1.00343809e+01, -1.22682428e+00, -1.94577579e+01,
        -2.76707697e+00, 1.47579327e+01,  4.94736385e+00,  2.68847847e+01,  -6.49254417e+00,
        1.94286156e+00,  -7.19223642e+00, -3.70413971e+00, -4.04303551e-01, -1.01827660e+01,
        1.49476433e+00};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
