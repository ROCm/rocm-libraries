/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(logsoftmax_test_axis_0)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.135261, -2.843968, -0.659995, -0.488413, -1.051857, -2.812936, -0.250956, -0.353985,
        -1.155980, -0.603651, -0.211969, -0.175371, -1.336552, -3.885010, -1.871544, -0.837083,
        -0.887745, -0.433338, -1.158864, -4.911197, -1.147972, -0.666711, -0.996874, -0.981418,
        -0.851145, -0.853988, -0.858112, -2.067420, -0.059956, -0.727436, -0.950881, -0.429689,
        -0.061906, -1.505332, -1.210277, -0.377970, -0.791448, -1.655428, -1.827253, -0.304828,
        -0.020762, -0.167101, -0.567346, -0.530319, -1.045094, -0.376648, -0.007391, -0.381670,
        -0.720302, -0.460499, -0.469651, -0.556740, -0.554628, -0.551582};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 0;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_1)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.550468, -2.132973, -1.549746, -0.650533, -1.051529, -2.248570, -0.141017, -2.028357,
        -1.947730, -1.511324, -0.166597, -0.379726, -1.965689, -1.172109, -1.475721, -2.700831,
        -1.537011, -0.658754, -1.596017, -3.353137, -2.266743, -1.084197, -1.076214, -0.406712,
        -2.743019, -0.425526, -1.079083, -2.139486, -1.270584, -1.024088, -1.154231, -3.201762,
        -0.888957, -0.532855, -3.103583, -1.221339, -1.355980, -3.531678, -1.438510, -0.975194,
        -0.080261, -1.162697, -1.568557, -1.398519, -1.322129, -0.470660, -0.370953, -0.907343,
        -1.179017, -3.312239, -1.286363, -1.586076, -0.345100, -0.824173};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_2)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.495957, -1.031212, -0.245531, -2.013726, -1.339125, -2.465619, -1.356652, -0.964037,
        -2.019250, -0.214522, -0.289569, -0.234392, -2.086591, -2.684439, -2.851651, -2.674176,
        -1.697424, -1.889155, -0.401029, -3.064586, -1.173030, -1.306912, -2.177020, -0.834262,
        -2.818177, -0.174415, -1.361105, -1.024571, -0.106766, -1.167645, -1.072650, -2.576522,
        -0.569261, -1.207483, -3.679894, -2.095913, -0.504264, -3.039291, -1.290559, -1.156812,
        -0.126453, -0.551493, -2.506384, -2.646261, -1.905195, -0.206994, -0.191369, -0.959754,
        -1.948685, -3.671233, -0.875521, -3.111952, -1.905644, -1.6076011};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 2;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, s));
}

TEST_CASE(logsoftmax_test_axis_3)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> a = {
        1.93885877,  -1.20006269, 0.90960855,  0.42108916,  -1.50797544, -1.31047913, 1.07816336,
        -1.13288733, -0.86411064, 0.97800238,  0.76631385,  2.07962834,  -0.8940665,  -1.62855592,
        -0.53763057, -1.48165117, -0.64154112, 0.42486547,  0.89330917,  -2.42022666, 0.192611,
        -0.01257413, -1.5326607,  0.53137897,  -1.52383859, 0.46994381,  0.00453619,  0.0066996,
        1.58394908,  0.84216752,  -0.04137941, -0.88580789, 1.44055158,  -0.17621241, -1.98917923,
        -0.08610038, 0.79020567,  -0.67714548, 0.42774631,  0.1376574,   2.23569227,  1.16681234,
        -1.21191456, -0.28411502, -0.18688975, 1.67552548,  2.48357974,  0.95891282,  -0.06616535,
        -0.99628491, 1.04314606,  -1.22943315, 0.76930403,  0.31106618};

    std::vector<float> s = {
        -0.336904, -3.475825, -1.366154, -0.279366, -2.208430, -2.010934, -0.225511, -2.436562,
        -2.167785, -1.572415, -1.784104, -0.470789, -1.067459, -1.801948, -0.711023, -2.307197,
        -1.467087, -0.400681, -0.426983, -3.740518, -1.127681, -1.078919, -2.599005, -0.534965,
        -2.561400, -0.567617, -1.033025, -2.097713, -0.520463, -1.262245, -1.763230, -2.607658,
        -0.281299, -0.814243, -2.627210, -0.724131, -0.655704, -2.123055, -1.018163, -2.480634,
        -0.382599, -1.451479, -1.843102, -0.915303, -0.818078, -1.316929, -0.508875, -2.033541,
        -1.487672, -2.417791, -0.378360, -2.568531, -0.569794, -1.028032};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 3, 3}};
    auto al  = mm->add_literal(migraphx::literal{a_shape, a});
    int axis = 3;
    mm->add_instruction(migraphx::make_op("logsoftmax", {{"axis", axis}}), al);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, s));
}
