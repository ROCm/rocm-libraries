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
#include <onnx_verify_utils.hpp>

TEST_CASE(skip_layer_normalization_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    std::vector<half> skip{half{1.2},
                           half{-1.0},
                           half{2.0},
                           half{1.0},
                           half{1.5},
                           half{2.2},
                           half{-3.3},
                           half{2.6},
                           half{1.0},
                           half{-10.0},
                           half{-1.0},
                           half{2.0},
                           half{-1.2},
                           half{1.6},
                           half{-1.1},
                           half{1.3}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};

    auto p = read_onnx("skip_layer_normalization_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_x, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());

    auto results                    = p.eval(pp);
    const auto& output              = results.at(0);
    const auto& mean                = results.at(1);
    const auto& inv_std_var         = results.at(2);
    const auto& input_skip_bias_sum = results.at(3);

    std::vector<half> result_vector;
    std::vector<half> mean_vector;
    std::vector<half> inv_std_var_vector;
    std::vector<half> input_skip_bias_sum_vector;

    output.visit([&](auto vals) { result_vector.assign(vals.begin(), vals.end()); });
    mean.visit([&](auto vals) { mean_vector.assign(vals.begin(), vals.end()); });
    inv_std_var.visit([&](auto vals) { inv_std_var_vector.assign(vals.begin(), vals.end()); });
    input_skip_bias_sum.visit(
        [&](auto vals) { input_skip_bias_sum_vector.assign(vals.begin(), vals.end()); });

    std::vector<half> gold      = {half{0.05770874},
                                   half{-0.34619141},
                                   half{2.30859375},
                                   half{-1.26953125},
                                   half{0.05160522},
                                   half{0.13891602},
                                   half{-6.91015625},
                                   half{-1.13476562},
                                   half{0.13244629},
                                   half{-0.29028320},
                                   half{-0.75732422},
                                   half{-0.69384766},
                                   half{-0.03375244},
                                   half{0.14160156},
                                   half{-5.88671875},
                                   half{-2.42187500}};
    std::vector<half> gold_mean = {
        half{1.12500000}, half{0.84960938}, half{0.50000000}, half{1.54882812}};
    std::vector<half> gold_inv_std_var = {
        half{0.65966797}, half{0.44873047}, half{0.12622070}, half{0.21801758}};
    std::vector<half> gold_input_skip_bias_sum = {half{2.00000000},
                                                  half{-1.50000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.39843750},
                                                  half{-3.00000000},
                                                  half{2.00000000},
                                                  half{11.00000000},
                                                  half{-11.00000000},
                                                  half{-1.00000000},
                                                  half{3.00000000},
                                                  half{0.00000000},
                                                  half{4.79687500},
                                                  half{-5.20312500},
                                                  half{6.60156250}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(mean_vector, gold_mean));
    EXPECT(migraphx::verify::verify_rms_range(inv_std_var_vector, gold_inv_std_var));
    EXPECT(
        migraphx::verify::verify_rms_range(input_skip_bias_sum_vector, gold_input_skip_bias_sum));
}

TEST_CASE(skip_layer_normalization_beta_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    std::vector<half> skip{half{1.2},
                           half{-1.0},
                           half{2.0},
                           half{1.0},
                           half{1.5},
                           half{2.2},
                           half{-3.3},
                           half{2.6},
                           half{1.0},
                           half{-10.0},
                           half{-1.0},
                           half{2.0},
                           half{-1.2},
                           half{1.6},
                           half{-1.1},
                           half{1.3}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};
    std::vector<half> beta{half{1.0}, half{1.0}, half{1.0}, half{1.0}};

    auto p = read_onnx("skip_layer_normalization_beta_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_x, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());
    pp["beta"]  = migraphx::argument(s_s, beta.data());

    auto results                    = p.eval(pp);
    const auto& output              = results.at(0);
    const auto& mean                = results.at(1);
    const auto& inv_std_var         = results.at(2);
    const auto& input_skip_bias_sum = results.at(3);

    std::vector<half> result_vector;
    std::vector<half> mean_vector;
    std::vector<half> inv_std_var_vector;
    std::vector<half> input_skip_bias_sum_vector;

    output.visit([&](auto vals) { result_vector.assign(vals.begin(), vals.end()); });
    mean.visit([&](auto vals) { mean_vector.assign(vals.begin(), vals.end()); });
    inv_std_var.visit([&](auto vals) { inv_std_var_vector.assign(vals.begin(), vals.end()); });
    input_skip_bias_sum.visit(
        [&](auto vals) { input_skip_bias_sum_vector.assign(vals.begin(), vals.end()); });

    std::vector<half> gold      = {half{1.05761719},
                                   half{0.65380859},
                                   half{3.30859375},
                                   half{-0.26953125},
                                   half{1.05175781},
                                   half{1.13867188},
                                   half{-5.91015625},
                                   half{-0.13476562},
                                   half{1.13281250},
                                   half{0.70996094},
                                   half{0.24267578},
                                   half{0.30615234},
                                   half{0.96630859},
                                   half{1.14160156},
                                   half{-4.88671875},
                                   half{-1.42187500}};
    std::vector<half> gold_mean = {
        half{1.12500000}, half{0.84960938}, half{0.50000000}, half{1.54882812}};
    std::vector<half> gold_inv_std_var = {
        half{0.65966797}, half{0.44873047}, half{0.12622070}, half{0.21801758}};
    std::vector<half> gold_input_skip_bias_sum = {half{2.00000000},
                                                  half{-1.50000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.39843750},
                                                  half{-3.00000000},
                                                  half{2.00000000},
                                                  half{11.00000000},
                                                  half{-11.00000000},
                                                  half{-1.00000000},
                                                  half{3.00000000},
                                                  half{0.00000000},
                                                  half{4.79687500},
                                                  half{-5.20312500},
                                                  half{6.60156250}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(mean_vector, gold_mean));
    EXPECT(migraphx::verify::verify_rms_range(inv_std_var_vector, gold_inv_std_var));
    EXPECT(
        migraphx::verify::verify_rms_range(input_skip_bias_sum_vector, gold_input_skip_bias_sum));
}

TEST_CASE(skip_layer_normalization_beta_bias_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    std::vector<half> skip{half{1.2},
                           half{-1.0},
                           half{2.0},
                           half{1.0},
                           half{1.5},
                           half{2.2},
                           half{-3.3},
                           half{2.6},
                           half{1.0},
                           half{-10.0},
                           half{-1.0},
                           half{2.0},
                           half{-1.2},
                           half{1.6},
                           half{-1.1},
                           half{1.3}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};
    std::vector<half> beta{half{1.0}, half{1.0}, half{1.0}, half{1.0}};
    std::vector<half> bias{half{1.0}, half{1.0}, half{1.0}, half{1.0}};

    auto p = read_onnx("skip_layer_normalization_beta_bias_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_x, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());
    pp["beta"]  = migraphx::argument(s_s, beta.data());
    pp["bias"]  = migraphx::argument(s_s, bias.data());

    auto results                    = p.eval(pp);
    const auto& output              = results.at(0);
    const auto& mean                = results.at(1);
    const auto& inv_std_var         = results.at(2);
    const auto& input_skip_bias_sum = results.at(3);

    std::vector<half> result_vector;
    std::vector<half> mean_vector;
    std::vector<half> inv_std_var_vector;
    std::vector<half> input_skip_bias_sum_vector;

    output.visit([&](auto vals) { result_vector.assign(vals.begin(), vals.end()); });
    mean.visit([&](auto vals) { mean_vector.assign(vals.begin(), vals.end()); });
    inv_std_var.visit([&](auto vals) { inv_std_var_vector.assign(vals.begin(), vals.end()); });
    input_skip_bias_sum.visit(
        [&](auto vals) { input_skip_bias_sum_vector.assign(vals.begin(), vals.end()); });

    std::vector<half> gold      = {half{1.05761719},
                                   half{0.65380859},
                                   half{3.30859375},
                                   half{-0.26953125},
                                   half{1.05175781},
                                   half{1.13867188},
                                   half{-5.91015625},
                                   half{-0.13476562},
                                   half{1.13281250},
                                   half{0.70996094},
                                   half{0.24267578},
                                   half{0.30615234},
                                   half{0.96630859},
                                   half{1.14160156},
                                   half{-4.88671875},
                                   half{-1.42187500}};
    std::vector<half> gold_mean = {
        half{2.12500000}, half{1.84960938}, half{1.50000000}, half{2.54882812}};
    std::vector<half> gold_inv_std_var = {
        half{0.65966797}, half{0.44873047}, half{0.12622070}, half{0.21801758}};
    std::vector<half> gold_input_skip_bias_sum = {half{3.00000000},
                                                  half{-0.50000000},
                                                  half{3.00000000},
                                                  half{3.00000000},
                                                  half{3.00000000},
                                                  half{3.39843750},
                                                  half{-2.00000000},
                                                  half{3.00000000},
                                                  half{12.00000000},
                                                  half{-10.00000000},
                                                  half{0.00000000},
                                                  half{4.00000000},
                                                  half{1.00000000},
                                                  half{5.79687500},
                                                  half{-4.20312500},
                                                  half{7.60156250}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(mean_vector, gold_mean));
    EXPECT(migraphx::verify::verify_rms_range(inv_std_var_vector, gold_inv_std_var));
    EXPECT(
        migraphx::verify::verify_rms_range(input_skip_bias_sum_vector, gold_input_skip_bias_sum));
}

TEST_CASE(skip_layer_normalization_2d_skip_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    // 2D skip has only 8 elements instead of 16
    std::vector<half> skip{
        half{1.2}, half{-1.0}, half{2.0}, half{1.0}, half{-1.2}, half{1.6}, half{-1.1}, half{1.3}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};

    auto p = read_onnx("skip_layer_normalization_2d_skip_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_skip{migraphx::shape::half_type, {2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_skip, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());

    auto results                    = p.eval(pp);
    const auto& output              = results.at(0);
    const auto& mean                = results.at(1);
    const auto& inv_std_var         = results.at(2);
    const auto& input_skip_bias_sum = results.at(3);

    std::vector<half> result_vector;
    std::vector<half> mean_vector;
    std::vector<half> inv_std_var_vector;
    std::vector<half> input_skip_bias_sum_vector;

    output.visit([&](auto vals) { result_vector.assign(vals.begin(), vals.end()); });
    mean.visit([&](auto vals) { mean_vector.assign(vals.begin(), vals.end()); });
    inv_std_var.visit([&](auto vals) { inv_std_var_vector.assign(vals.begin(), vals.end()); });
    input_skip_bias_sum.visit(
        [&](auto vals) { input_skip_bias_sum_vector.assign(vals.begin(), vals.end()); });

    std::vector<half> gold = {half{0.0577393},
                              half{-0.346191},
                              half{2.3125},
                              half{-1.27051},
                              half{-0.0884399},
                              half{0.288574},
                              half{-3.91406},
                              half{-0.922363},
                              half{0.162842},
                              half{-0.218628},
                              half{-1.07227},
                              half{0.589355},
                              half{-0.0338135},
                              half{0.141846},
                              half{-5.89062},
                              half{-2.42188}};

    std::vector<half> gold_mean = {half{1.12402}, half{0.25}, half{3.29883}, half{1.54883}};

    std::vector<half> gold_inv_std_var = {
        half{0.660156}, half{0.932129}, half{0.206543}, half{0.218506}};

    std::vector<half> gold_input_skip_bias_sum = {half{1.99902},
                                                  half{-1.5},
                                                  half{2},
                                                  half{2},
                                                  half{-0.699219},
                                                  half{1.79883},
                                                  half{-0.799805},
                                                  half{0.700195},
                                                  half{11.1953},
                                                  half{-2},
                                                  half{2},
                                                  half{2},
                                                  half{0},
                                                  half{4.79688},
                                                  half{-5.19531},
                                                  half{6.59375}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(mean_vector, gold_mean));
    EXPECT(migraphx::verify::verify_rms_range(inv_std_var_vector, gold_inv_std_var));
    EXPECT(
        migraphx::verify::verify_rms_range(input_skip_bias_sum_vector, gold_input_skip_bias_sum));
}

TEST_CASE(skip_layer_normalization_skip_batch_size_1_test)
{
    using migraphx::half;
    std::vector<half> x{half{0.8},
                        half{-0.5},
                        half{0.0},
                        half{1.0},
                        half{0.5},
                        half{0.2},
                        half{0.3},
                        half{-0.6},
                        half{10.0},
                        half{-1.0},
                        half{0.0},
                        half{1.0},
                        half{1.2},
                        half{3.2},
                        half{-4.1},
                        half{5.3}};
    std::vector<half> skip{
        half{1.2}, half{-1.0}, half{2.0}, half{1.0}, half{1.5}, half{2.2}, half{-3.3}, half{2.6}};
    std::vector<half> scale{half{0.1}, half{0.2}, half{4.0}, half{-2.2}};

    auto p = read_onnx("skip_layer_normalization_skip_batch_size_1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::half_type, {2, 2, 4}};
    migraphx::shape s_skip{migraphx::shape::half_type, {1, 2, 4}};
    migraphx::shape s_s{migraphx::shape::half_type, {4}};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["skip"]  = migraphx::argument(s_skip, skip.data());
    pp["gamma"] = migraphx::argument(s_s, scale.data());

    auto results                    = p.eval(pp);
    const auto& output              = results.at(0);
    const auto& mean                = results.at(1);
    const auto& inv_std_var         = results.at(2);
    const auto& input_skip_bias_sum = results.at(3);

    std::vector<half> result_vector;
    std::vector<half> mean_vector;
    std::vector<half> inv_std_var_vector;
    std::vector<half> input_skip_bias_sum_vector;

    output.visit([&](auto vals) { result_vector.assign(vals.begin(), vals.end()); });
    mean.visit([&](auto vals) { mean_vector.assign(vals.begin(), vals.end()); });
    inv_std_var.visit([&](auto vals) { inv_std_var_vector.assign(vals.begin(), vals.end()); });
    input_skip_bias_sum.visit(
        [&](auto vals) { input_skip_bias_sum_vector.assign(vals.begin(), vals.end()); });

    std::vector<half> gold = {half{0.05773491},
                              half{-0.34640941},
                              half{2.30939627},
                              half{-1.27016795},
                              half{0.05159748},
                              half{0.13908887},
                              half{-6.90957546},
                              half{-1.13514447},
                              half{0.16306864},
                              half{-0.21880098},
                              half{-1.07336318},
                              half{0.59034979},
                              half{0.00946274},
                              half{0.11183234},
                              half{-6.57230043},
                              half{-2.17642951}};

    std::vector<half> gold_mean = {
        half{1.12500000}, half{0.85000002}, half{3.29999995}, half{2.15000010}};

    std::vector<half> gold_inv_std_var = {
        half{0.65982747}, half{0.44867375}, half{0.20641601}, half{0.17204976}};

    std::vector<half> gold_input_skip_bias_sum = {half{2.00000000},
                                                  half{-1.50000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.40000010},
                                                  half{-3.00000000},
                                                  half{1.99999988},
                                                  half{11.19999981},
                                                  half{-2.00000000},
                                                  half{2.00000000},
                                                  half{2.00000000},
                                                  half{2.70000005},
                                                  half{5.40000010},
                                                  half{-7.39999962},
                                                  half{7.90000010}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    EXPECT(migraphx::verify::verify_rms_range(mean_vector, gold_mean));
    EXPECT(migraphx::verify::verify_rms_range(inv_std_var_vector, gold_inv_std_var));
    EXPECT(
        migraphx::verify::verify_rms_range(input_skip_bias_sum_vector, gold_input_skip_bias_sum));
}
