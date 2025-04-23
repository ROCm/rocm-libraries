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
#include <iostream>
#include <migraphx/program.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

/*!
 * Example MIGraphX programs for following the Contributor's Guide.
 */

TEST_CASE(add_two_literals)
{
    /*!
     * Simple MIGraphX program to add two literal values.
     * Equivalent to adding two constant scalar values together.
     */
    // create the program a get a pointer to the main module
    migraphx::program p;
    auto* mm = p.get_main_module();

    // add two literals to the program
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);

    // make the "add" operation between the two literals and add it to the program
    mm->add_instruction(migraphx::make_op("add"), one, two);

    // compile the program on the reference device
    p.compile(migraphx::make_target("ref"));

    // evaulate the program and retreive the result
    auto result = p.eval({}).back();
    std::cout << "add_two_literals: 1 + 2 = " << result << "\n";
    EXPECT(result.at<int>() == 3);
}

TEST_CASE(add_parameters)
{
    /*!
     * Modified version of MIGraphX program seen in add_two_literals to accept a parameter.
     * Equivalent to adding a constant scalar value with another scalar input.
     */
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1}};

    // add a "x" parameter with the shape s
    auto x   = mm->add_parameter("x", s);
    auto two = mm->add_literal(2);

    // add the "add" instruction between the "x" parameter and "two" to the module
    mm->add_instruction(migraphx::make_op("add"), x, two);
    p.compile(migraphx::make_target("ref"));

    // create a parameter_map object for passing a value to the "x" parameter
    std::vector<int> data = {4};
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(params).back();
    std::cout << "add_parameters: 4 + 2 = " << result << "\n";
    EXPECT(result.at<int>() == 6);
}

TEST_CASE(handling_tensors)
{
    /*!
     * This example does a convolution operation over an input tensor using the given weighting
     * tensor. This is meant to show an example of working with tensors in MIGraphX. The output
     * tensor is compared against a precomputed solution tensor at the end of the program.
     */
    migraphx::program p;
    auto* mm = p.get_main_module();

    // create shape objects for the input tensor and weights
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {2, 3, 3, 3}};

    // create the parameters and add the "convolution" operation to the module
    auto input   = mm->add_parameter("X", input_shape);
    auto weights = mm->add_parameter("W", weights_shape);
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}),
                        input,
                        weights);

    p.compile(migraphx::make_target("ref"));

    // Allocated buffers by the user
    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    // Create the arguments in a parameter_map
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_shape, a.data());
    params["W"] = migraphx::argument(weights_shape, c.data());

    // Evaluate and confirm the result
    auto result = p.eval(params).back();
    std::vector<float> results_vector(64);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // Solution vector
    std::vector<float> gold = {-0.20817225,
                               0.87965256,
                               0.14958936,
                               -1.24887264,
                               -0.06540672,
                               0.20778663,
                               0.40456355,
                               -0.99900877,
                               0.4917807,
                               0.1994698,
                               0.64205718,
                               0.37798831,
                               -0.25315839,
                               0.44276932,
                               -0.16138598,
                               0.79344082};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
