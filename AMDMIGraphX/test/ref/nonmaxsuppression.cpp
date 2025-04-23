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

TEST_CASE(nms_dyn_out_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_l,
        scores_l,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_dyn_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 3}, {6, 6}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 3}, {1, 1}, {6, 6}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                     0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0,
                                     0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                     0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {2, 1, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 1, 0, 3, 1, 0, 0, 1, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_dyn_boxes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 1}, {4, 20}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 1}, {1, 1}, {4, 20}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                     0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_dyn_classes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {{1, 1}, {6, 6}, {4, 4}}};

    migraphx::shape scores_s{migraphx::shape::float_type, {{1, 1}, {1, 3}, {6, 6}}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{2});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression",
                          {{"center_point_box", true}, {"use_dyn_output", true}}),
        boxes_p,
        scores_p,
        max_out_l,
        iou_threshold,
        score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> boxes_vec  = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape input_fixed_shape1{migraphx::shape::float_type, {1, 2, 6}};
    migraphx::parameter_map params0;
    params0["boxes"]  = migraphx::argument(input_fixed_shape0, boxes_vec.data());
    params0["scores"] = migraphx::argument(input_fixed_shape1, scores_vec.data());
    auto output       = p.eval(params0).back();

    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_not_center_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                    1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    // set use_dyn_output back to false in operator map
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"use_dyn_output", false}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_transpose1_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 4, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), t_boxes_l);
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(nms_transpose2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {4, 1, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t_boxes_l);
    auto r =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    mm->add_return({r});

    p.compile(migraphx::make_target("ref"));
    auto output = p.eval({}).back();
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}
