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

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_ones)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f, 1.38629f, 1.38629f, 1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_asym_test)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_asym_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> score_data = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    migraphx::shape label_shape{migraphx::shape::int32_type, {3}};
    std::vector<int32_t> label_data = {0, 3, 1};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 1.38629436, 1.38629436};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_one_hot_test)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {1, 1, 1, 1};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.74366838, 0.74366838, 1.74366838, 1.74366838};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_zeros)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 0);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f, 1.38629f, 1.38629f, 1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test_ones)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.5452f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test_zeroes)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 0);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.5452f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_test_ones)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_test_zeroes)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 0);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_rand_data)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data{0.96771913,
                                  0.41032661,
                                  0.20715942,
                                  0.03879957,
                                  0.72009297,
                                  0.87246912,
                                  0.45548323,
                                  0.08531375,
                                  0.19684932,
                                  0.53406917,
                                  0.96924279,
                                  0.32202707,
                                  0.41944426,
                                  0.42816934,
                                  0.36742527,
                                  0.6424184};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 2, 1, 0};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.88998183, 1.50650129, 1.40313031, 1.43695476};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_test_rand_data)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data{0.70832104,
                                  0.57696298,
                                  0.60347884,
                                  0.93392199,
                                  0.25057635,
                                  0.33402099,
                                  0.57135317,
                                  0.38481569,
                                  0.14061437,
                                  0.60612205,
                                  0.24493107,
                                  0.07951325,
                                  0.32728171,
                                  0.06750434,
                                  0.34526496,
                                  0.28548175};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 2, 1, 0};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.2480964271388393};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test_rand_data)
{
    migraphx::program p = optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data{0.49182203,
                                  0.68737944,
                                  0.98275259,
                                  0.52598715,
                                  0.28430275,
                                  0.59839527,
                                  0.84222958,
                                  0.19842379,
                                  0.5450379,
                                  0.58118435,
                                  0.40838571,
                                  0.55503269,
                                  0.9424222,
                                  0.38791064,
                                  0.74781717,
                                  0.37622127};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 2, 1, 0};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.060988893272207};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
