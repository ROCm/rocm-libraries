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

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_even_weighted_ones_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 1.38629436, 1.38629436, 1.38629436};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ones_asym_batch)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_asym_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {3}};
    std::vector<int32_t> label_data = {0, 3, 1};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ones_asym_class)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_asym_class_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 3}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 2, 1, 0};
    migraphx::shape weight_shape{migraphx::shape::float_type, {3}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.09861229, 2.19722458, 0.54930614, 1.09861229};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ones_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 2.77258872};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ones_out_bounds_ignore_idx_test)
{
    migraphx::program p = optimize_onnx(
        "softmaxcrossentropyloss_2d_no_reduction_weighted_out_bounds_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 2.77258872};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(
    softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ones_neg_out_bounds_ignore_idx_test)
{
    migraphx::program p = optimize_onnx(
        "softmaxcrossentropyloss_2d_no_reduction_weighted_neg_out_bounds_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 2.77258872};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ignore_index_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 0.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_ignore_index_iota_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::iota(score_data.begin(), score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {3.4401897f, 1.3205691f, 1.22009485f, 0.f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_weighted_iota_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::iota(score_data.begin(), score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {8.861233040647768};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_weighted_iota_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::iota(score_data.begin(), score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.3632666216381182};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_weighted_iota_zero_one_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::fill(score_data.begin(), score_data.begin() + 4, 0);
    std::fill(score_data.begin() + 4, score_data.begin() + 8, 1);
    std::iota(score_data.begin() + 8, score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {1, 1, 1, 1};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.69314718f, 0.69314718f, 1.22009485f, 1.22009485f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_weighted_iota_zero_one_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::fill(score_data.begin(), score_data.begin() + 4, 0);
    std::fill(score_data.begin() + 4, score_data.begin() + 8, 1);
    std::iota(score_data.begin() + 8, score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {1, 1, 1, 1};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {3.8264840596810856f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_weighted_iota_zero_one_data_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements());
    std::fill(score_data.begin(), score_data.begin() + 4, 0);
    std::fill(score_data.begin() + 4, score_data.begin() + 8, 1);
    std::iota(score_data.begin() + 8, score_data.end(), 1);

    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {1, 1, 1, 1};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.9132420298405428f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_uneven_weighted_neg_ignore_index_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 2.77259f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_index_and_label_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, -2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0.69314718, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_index_and_label_test2)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test2.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, -1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436, 4.15888308, 0, 2.77259f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_neg_ignore_index_invalid_label_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_no_reduction_weighted_neg_ignore_idx_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, -5};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    // Should throw as even through ignore_idx out of bounds and negative label being out of bounds
    // is invalid
    EXPECT(test::throws([&] { std::ignore = p.eval(pp).back(); }));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_weighted_test_ones)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.545177};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_weighted_test_zeroes)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_sum_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 0.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.545177};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_weighted_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_uneven_weighted_test)
{
    migraphx::program p =
        optimize_onnx("softmaxcrossentropyloss_2d_mean_reduction_weighted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data(score_shape.elements(), 1.0f);
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};
    migraphx::shape weight_shape{migraphx::shape::float_type, {4}};
    std::vector<float> weight_data = {1.0f, 0.5f, 2.0f, 3.0f};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());
    pp["2"] = migraphx::argument(weight_shape, weight_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629436};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
