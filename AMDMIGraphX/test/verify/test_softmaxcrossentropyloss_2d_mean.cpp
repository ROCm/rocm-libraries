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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t DType,
          migraphx::shape::type_t LType,
          const size_t NumClasses,
          const size_t NumBatches>
struct test_softmaxcrossentropyloss_2d_mean
    : verify_program<test_softmaxcrossentropyloss_2d_mean<DType, LType, NumClasses, NumBatches>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        size_t batch_size = NumBatches;
        size_t class_size = NumClasses;

        auto scores = mm->add_parameter("0", migraphx::shape{DType, {NumBatches, NumClasses}});
        auto labels =
            mm->add_literal(migraphx::literal(migraphx::shape(LType, {batch_size}), {0, 1, 2, 3}));

        auto weights = mm->add_literal(migraphx::literal(migraphx::shape(DType, {1}, {0}), {1}));

        std::vector<size_t> label_indexes(NumBatches);
        std::iota(label_indexes.begin(), label_indexes.end(), 0);

        auto labels_idx = mm->add_literal(
            migraphx::literal(migraphx::shape(LType, {NumBatches}, {1}), label_indexes));

        auto mb_weights = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {batch_size}}}), weights);

        auto softmax = mm->add_instruction(migraphx::make_op("softmax"), scores);

        auto unsq_labels =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {-1}}}), labels);
        auto unsq_labels_idx =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), labels_idx);
        auto bc_unsq_labels_idx = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, 1}}}), unsq_labels_idx);
        auto concat = mm->add_instruction(
            migraphx::make_op("concat", {{"axis", -1}}), bc_unsq_labels_idx, unsq_labels);
        auto gathernd = mm->add_instruction(migraphx::make_op("gathernd"), softmax, concat);
        auto unsq_mb_weights =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), mb_weights);
        auto unsq_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, class_size}}}),
            unsq_mb_weights);
        auto gathernd2 = mm->add_instruction(migraphx::make_op("gathernd"), unsq_mb, concat);

        auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), gathernd);
        auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

        auto weighted_loss =
            mm->add_instruction(migraphx::make_op("mul"), neglogsoftmax, gathernd2);
        mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0}}}), weighted_loss);

        return p;
    }

    std::string section() const { return "reduce"; }
};

// template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::double_type,
// migraphx::shape::int32_type, 4, 4>; template struct
// test_softmaxcrossentropyloss_2d_mean<migraphx::shape::double_type, migraphx::shape::int64_type,
// 4, 4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::float_type,
                                                     migraphx::shape::int32_type,
                                                     4,
                                                     4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::float_type,
                                                     migraphx::shape::int64_type,
                                                     4,
                                                     4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::half_type,
                                                     migraphx::shape::int32_type,
                                                     4,
                                                     4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::half_type,
                                                     migraphx::shape::int64_type,
                                                     4,
                                                     4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::bf16_type,
                                                     migraphx::shape::int32_type,
                                                     4,
                                                     4>;
template struct test_softmaxcrossentropyloss_2d_mean<migraphx::shape::bf16_type,
                                                     migraphx::shape::int64_type,
                                                     4,
                                                     4>;
