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

#include <onnx_test.hpp>

TEST_CASE(softmaxcrossentropyloss_kd_sum_reduction_weighted_double_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::double_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {class_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::double_type, {class_size}});

    auto weights_dflt = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::double_type, {1}, {0}), {1}));
    auto labels_idx = mm->add_literal(migraphx::literal(
        migraphx::shape(migraphx::shape::int32_type, {class_size}, {1}), {0, 1, 2, 3}));

    // Index variables used for gather on k dimensions that span their dimension
    auto kd_1 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));
    auto kd_2 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));

    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {class_size}}}),
                        weights_dflt);
    weights = mm->add_instruction(migraphx::make_op("neg"), weights);

    auto softmax = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto unsq_labels =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {-1}}}), labels);

    auto unsq_labels_idx =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), labels_idx);
    auto bc_unsq_labels_idx = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx);

    auto unsq_labels_idx2 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), kd_1);
    auto bc_unsq_labels_idx2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx2);

    auto unsq_labels_idx3 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 3}}}), kd_2);
    auto bc_unsq_labels_idx3 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx3);

    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}),
                                      bc_unsq_labels_idx,
                                      bc_unsq_labels_idx2,
                                      bc_unsq_labels_idx3,
                                      unsq_labels);

    auto transpose = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), softmax);

    auto gathernd = mm->add_instruction(migraphx::make_op("gathernd"), transpose, concat);
    auto unsq_mb_weights =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), weights);
    auto unsq_mb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", scores->get_shape().lens()}}),
        unsq_mb_weights);
    auto transpose2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq_mb);
    auto gathernd2 = mm->add_instruction(migraphx::make_op("gathernd"), transpose2, concat);

    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), gathernd);
    auto weighted_loss = mm->add_instruction(migraphx::make_op("mul"), logsoftmax, gathernd2);
    mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2}}}), weighted_loss);

    auto prog = optimize_onnx("softmaxcrossentropyloss_kd_sum_reduction_double_weighted_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_kd_no_reduction_weighted_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {class_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {class_size}});

    auto weights_dflt = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::float_type, {1}, {0}), {1}));
    auto labels_idx = mm->add_literal(migraphx::literal(
        migraphx::shape(migraphx::shape::int32_type, {class_size}, {1}), {0, 1, 2, 3}));

    // Index variables used for gather on k dimensions that span their dimension
    auto kd_1 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));
    auto kd_2 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));

    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {class_size}}}),
                        weights_dflt);
    weights = mm->add_instruction(migraphx::make_op("neg"), weights);

    auto softmax = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto unsq_labels =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {-1}}}), labels);

    auto unsq_labels_idx =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), labels_idx);
    auto bc_unsq_labels_idx = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx);

    auto unsq_labels_idx2 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), kd_1);
    auto bc_unsq_labels_idx2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx2);

    auto unsq_labels_idx3 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 3}}}), kd_2);
    auto bc_unsq_labels_idx3 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx3);

    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}),
                                      bc_unsq_labels_idx,
                                      bc_unsq_labels_idx2,
                                      bc_unsq_labels_idx3,
                                      unsq_labels);

    auto transpose = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), softmax);

    auto gathernd = mm->add_instruction(migraphx::make_op("gathernd"), transpose, concat);
    auto unsq_mb_weights =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), weights);
    auto unsq_mb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", scores->get_shape().lens()}}),
        unsq_mb_weights);
    auto transpose2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq_mb);
    auto gathernd2 = mm->add_instruction(migraphx::make_op("gathernd"), transpose2, concat);

    auto logsoftmax = mm->add_instruction(migraphx::make_op("log"), gathernd);
    mm->add_instruction(migraphx::make_op("mul"), logsoftmax, gathernd2);

    auto prog = optimize_onnx("softmaxcrossentropyloss_kd_no_reduction_weighted_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_kd_mean_reduction_half_weighted_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::half_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {class_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::half_type, {class_size}});

    auto weights_dflt = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::half_type, {1}, {0}), {1}));
    auto labels_idx = mm->add_literal(migraphx::literal(
        migraphx::shape(migraphx::shape::int32_type, {class_size}, {1}), {0, 1, 2, 3}));

    // Index variables used for gather on k dimensions that span their dimension
    auto kd_1 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));
    auto kd_2 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));

    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {class_size}}}),
                        weights_dflt);
    weights = mm->add_instruction(migraphx::make_op("neg"), weights);

    auto softmax = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto unsq_labels =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {-1}}}), labels);

    auto unsq_labels_idx =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), labels_idx);
    auto bc_unsq_labels_idx = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx);

    auto unsq_labels_idx2 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), kd_1);
    auto bc_unsq_labels_idx2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx2);

    auto unsq_labels_idx3 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 3}}}), kd_2);
    auto bc_unsq_labels_idx3 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx3);

    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}),
                                      bc_unsq_labels_idx,
                                      bc_unsq_labels_idx2,
                                      bc_unsq_labels_idx3,
                                      unsq_labels);

    auto transpose = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), softmax);

    auto gathernd = mm->add_instruction(migraphx::make_op("gathernd"), transpose, concat);
    auto unsq_mb_weights =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), weights);
    auto unsq_mb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", scores->get_shape().lens()}}),
        unsq_mb_weights);
    auto transpose2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq_mb);
    auto gathernd2 = mm->add_instruction(migraphx::make_op("gathernd"), transpose2, concat);

    auto logsoftmax = mm->add_instruction(migraphx::make_op("log"), gathernd);

    auto weighted_loss = mm->add_instruction(migraphx::make_op("mul"), logsoftmax, gathernd2);

    auto loss_x =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2}}}), weighted_loss);
    auto loss_w =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2}}}), gathernd2);
    loss_w = mm->add_instruction(migraphx::make_op("neg"), loss_w);

    mm->add_instruction(migraphx::make_op("div"), loss_x, loss_w);

    auto prog = optimize_onnx("softmaxcrossentropyloss_kd_mean_reduction_half_weighted_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_kd_mean_reduction_bf16_weighted_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::bf16_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {class_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::bf16_type, {class_size}});

    auto weights_dflt = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::bf16_type, {1}, {0}), {1}));
    auto labels_idx = mm->add_literal(migraphx::literal(
        migraphx::shape(migraphx::shape::int32_type, {class_size}, {1}), {0, 1, 2, 3}));

    // Index variables used for gather on k dimensions that span their dimension
    auto kd_1 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));
    auto kd_2 = mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int32_type, {2}, {1}), {0, 1}));

    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {class_size}}}),
                        weights_dflt);
    weights = mm->add_instruction(migraphx::make_op("neg"), weights);

    auto softmax = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto unsq_labels =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {-1}}}), labels);

    auto unsq_labels_idx =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3}}}), labels_idx);
    auto bc_unsq_labels_idx = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx);

    auto unsq_labels_idx2 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), kd_1);
    auto bc_unsq_labels_idx2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx2);

    auto unsq_labels_idx3 =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1, 3}}}), kd_2);
    auto bc_unsq_labels_idx3 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", unsq_labels->get_shape().lens()}}),
        unsq_labels_idx3);

    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", -1}}),
                                      bc_unsq_labels_idx,
                                      bc_unsq_labels_idx2,
                                      bc_unsq_labels_idx3,
                                      unsq_labels);

    auto transpose = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), softmax);

    auto gathernd = mm->add_instruction(migraphx::make_op("gathernd"), transpose, concat);
    auto unsq_mb_weights =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), weights);
    auto unsq_mb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", scores->get_shape().lens()}}),
        unsq_mb_weights);
    auto transpose2 = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), unsq_mb);
    auto gathernd2 = mm->add_instruction(migraphx::make_op("gathernd"), transpose2, concat);

    auto logsoftmax = mm->add_instruction(migraphx::make_op("log"), gathernd);

    auto weighted_loss = mm->add_instruction(migraphx::make_op("mul"), logsoftmax, gathernd2);

    auto loss_x =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2}}}), weighted_loss);
    auto loss_w =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2}}}), gathernd2);
    loss_w = mm->add_instruction(migraphx::make_op("neg"), loss_w);

    mm->add_instruction(migraphx::make_op("div"), loss_x, loss_w);

    auto prog = optimize_onnx("softmaxcrossentropyloss_kd_mean_reduction_bf16_weighted_test.onnx");
    EXPECT(p == prog);
}
