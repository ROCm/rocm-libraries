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
#include <onnx_verify_utils.hpp>

TEST_CASE(negativeloglikelihoodloss_kd_no_reduction_weighted_test)
{
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_no_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4, 2, 2}};
    std::vector<float> score_data(score_shape.elements(), 0.0f);
    std::iota(score_data.begin(), score_data.end(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4, 2, 2}};
    std::vector<int32_t> label_data = {0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-1.0f,
                               -56.0f,
                               -3.5f,
                               -24.0f,
                               -17.0f,
                               -120.0f,
                               -11.5f,
                               -56.0f,
                               -33.0f,
                               -184.0f,
                               -19.5f,
                               -88.0f,
                               -49.0f,
                               -248.0f,
                               -27.5f,
                               -120.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(negativeloglikelihoodloss_kd_no_reduction_weighted_test2)
{
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_no_reduction_weighted_test2.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {2, 3, 2}};
    std::vector<float> score_data = {
        1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 2.0f, 0.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f};

    migraphx::shape label_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> label_data = {2, 1, 0, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {3}};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-3.0f, -2.0f, 0.0f, -2.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(negativeloglikelihoodloss_kd_sum_reduction_weighted_test)
{
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_sum_reduction_double_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::double_type, {4, 4, 2, 2}};
    std::vector<double> score_data(score_shape.elements());
    std::iota(score_data.begin(), score_data.end(), 1.0);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4, 2, 2}};
    std::vector<int32_t> label_data = {0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::double_type, {4}};
    std::vector<double> weight_data = {1.0, 0.5, 2.0, 4.0};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<double> gold = {-1058.0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(negativeloglikelihoodloss_kd_sum_reduction_weighted_test2)
{
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_sum_reduction_double_weighted_test2.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::double_type, {2, 3, 2}};
    std::vector<double> score_data = {1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0};

    migraphx::shape label_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> label_data = {2, 1, 0, 2};
    migraphx::shape weight_shape{migraphx::shape::double_type, {3}};
    std::vector<double> weight_data = {0.2, 0.3, 0.1};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<double> gold = {-1.1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(negativeloglikelihoodloss_kd_mean_reduction_weighted_test)
{
    using migraphx::half;
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_mean_reduction_half_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::half_type, {4, 4, 2, 2}};
    std::vector<half> score_data(score_shape.elements());
    std::iota(score_data.begin(), score_data.end(), half(1.0));

    migraphx::shape label_shape{migraphx::shape::int32_type, {4, 2, 2}};
    std::vector<int32_t> label_data = {0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::half_type, {4}};
    std::vector<double> weight_data = {half(1.0), half(0.5), half(2.0), half(4.0)};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold = {half{-35.266666666666666}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(negativeloglikelihoodloss_kd_mean_reduction_weighted_test2)
{
    using migraphx::half;
    migraphx::program p =
        optimize_onnx("negativeloglikelihoodloss_kd_mean_reduction_half_weighted_test2.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::half_type, {2, 3, 2}};
    std::vector<half> score_data = {half(1.0),
                                    half(2.0),
                                    half(2.0),
                                    half(2.0),
                                    half(3.0),
                                    half(2.0),
                                    half(0.0),
                                    half(1.0),
                                    half(2.0),
                                    half(2.0),
                                    half(1.0),
                                    half(2.0)};

    migraphx::shape label_shape{migraphx::shape::int32_type, {2, 2}};
    std::vector<int32_t> label_data = {2, 1, 0, 2};
    migraphx::shape weight_shape{migraphx::shape::half_type, {3}};
    std::vector<half> weight_data = {half(0.2), half(0.3), half(0.1)};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold = {half{-1.5714285714285714}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
