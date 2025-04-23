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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/common.hpp>
#include <onnx_test.hpp>

TEST_CASE(quant_convolution_dual_zero_bias_test)
{
    // TODO: use other dual_bias test, verify with other framework once convinteger supported
    migraphx::program p = read_onnx("convinteger_dual_bias_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::int8_type, {1, 3, 5, 5}};
    std::vector<int8_t> data_a = {
        0,    -2,  4,    -6,  8,    -10,  12,  -14,  16,  -18,  20,  22,   -24, 26,   -28,
        30,   -32, 34,   -36, 38,   -40,  42,  -44,  46,  -48,  50,  -52,  54,  -56,  58,
        -60,  62,  -64,  66,  -68,  70,   -72, 74,   -76, 78,   -80, 82,   -84, 86,   -88,
        90,   -92, 94,   -96, 98,   -100, 102, -104, 106, -108, 110, -112, 114, -116, 118,
        -120, 122, -124, 126, -127, 127,  -64, 32,   -16, 8,    -4,  2,    -1,  0,    1};

    migraphx::shape b{migraphx::shape::int8_type, {1, 3, 2, 2}};
    std::vector<int8_t> data_b = {-127, -64, -32, -8, -4, 0, 2, 4, 8, 16, 64, 127};

    migraphx::shape a_bias{migraphx::shape::int8_type, {1}, {1}};
    std::vector<int8_t> data_a_bias = {0};

    migraphx::shape b_bias{migraphx::shape::int8_type, {1}, {1}};
    std::vector<int8_t> data_b_bias = {0};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(a, data_a.data());
    pp["1"] = migraphx::argument(b, data_b.data());
    pp["2"] = migraphx::argument(a_bias, data_a_bias.data());
    pp["3"] = migraphx::argument(b_bias, data_b_bias.data());

    auto result = p.eval(pp).back();

    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {-6072,
                                 6264,
                                 -6456,
                                 6648,
                                 6680,
                                 -8248,
                                 8536,
                                 -8697,
                                 -3772,
                                 -1430,
                                 1504,
                                 -1570,
                                 -696,
                                 761,
                                 -898,
                                 1035};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(quant_convolution_dual_non_zero_bias_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
    migraphx::program p = read_onnx("convinteger_dual_bias_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::int8_type, {1, 3, 5, 5}};
    std::vector<int8_t> data_a = {
        0,    -2,  4,    -6,  8,    -10,  12,  -14,  16,  -18,  20,  22,   -24, 26,   -28,
        30,   -32, 34,   -36, 38,   -40,  42,  -44,  46,  -48,  50,  -52,  54,  -56,  58,
        -60,  62,  -64,  66,  -68,  70,   -72, 74,   -76, 78,   -80, 82,   -84, 86,   -88,
        90,   -92, 94,   -96, 98,   -100, 102, -104, 106, -108, 110, -112, 114, -116, 118,
        -120, 122, -124, 126, -127, 127,  -64, 32,   -16, 8,    -4,  2,    -1,  0,    1};

    migraphx::shape b{migraphx::shape::int8_type, {1, 3, 2, 2}};
    std::vector<int8_t> data_b = {-127, -64, -32, -8, -4, 0, 2, 4, 8, 16, 64, 127};

    migraphx::shape a_bias{migraphx::shape::int8_type, {1}, {1}};
    std::vector<int8_t> data_a_bias = {10};

    migraphx::shape b_bias{migraphx::shape::int8_type, {1}, {1}};
    std::vector<int8_t> data_b_bias = {-2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(a, data_a.data());
    pp["1"] = migraphx::argument(b, data_b.data());
    pp["2"] = migraphx::argument(a_bias, data_a_bias.data());
    pp["3"] = migraphx::argument(b_bias, data_b_bias.data());

    auto result = p.eval(pp).back();

    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // create the following program to compare:
    // conv(x-x_bias,w-w_bias)
    // where datatypes for x,w,x_bias,w_bias are int32
    migraphx::program p2;
    migraphx::module* mm = p2.get_main_module();

    migraphx::shape a_i32{migraphx::shape::int32_type, {1, 3, 5, 5}};
    migraphx::shape b_i32{migraphx::shape::int32_type, {1, 3, 2, 2}};

    migraphx::shape bias_i32{migraphx::shape::int32_type, {1}, {1}};
    auto x            = mm->add_parameter("0", a_i32);
    auto weights      = mm->add_parameter("1", b_i32);
    auto x_bias       = mm->add_parameter("2", bias_i32);
    auto weights_bias = mm->add_parameter("3", bias_i32);

    auto sub_input   = add_common_op(*mm, migraphx::make_op("sub"), {x, x_bias});
    auto sub_weights = add_common_op(*mm, migraphx::make_op("sub"), {weights, weights_bias});
    mm->add_instruction(migraphx::make_op("convolution"), sub_input, sub_weights);

    std::vector<int32_t> data_a_i32(data_a.begin(), data_a.end());
    std::vector<int32_t> data_b_i32(data_b.begin(), data_b.end());
    std::vector<int32_t> data_a_bias_i32 = {10};
    std::vector<int32_t> data_b_bias_i32 = {-2};

    migraphx::parameter_map pp2;
    pp2["0"] = migraphx::argument(a_i32, data_a_i32.data());
    pp2["1"] = migraphx::argument(b_i32, data_b_i32.data());
    pp2["2"] = migraphx::argument(bias_i32, data_a_bias_i32.data());
    pp2["3"] = migraphx::argument(bias_i32, data_b_bias_i32.data());

    auto result2 = p2.eval(pp2).back();

    std::vector<int32_t> result_vector_i32;
    result2.visit([&](auto output) { result_vector_i32.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(result_vector, result_vector_i32));
}
