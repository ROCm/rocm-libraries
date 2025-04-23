/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

struct test_nms : verify_program<test_nms>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};

        migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
        std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

        auto boxes_l         = mm->add_parameter("boxes", boxes_s);
        auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
        auto max_out_l       = mm->add_literal(int64_t{4});
        auto iou_threshold   = mm->add_literal(0.5f);
        auto score_threshold = mm->add_literal(0.0f);

        auto r =
            mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", 1}}),
                                boxes_l,
                                scores_l,
                                max_out_l,
                                iou_threshold,
                                score_threshold);
        mm->add_return({r});

        return p;
    }
};
