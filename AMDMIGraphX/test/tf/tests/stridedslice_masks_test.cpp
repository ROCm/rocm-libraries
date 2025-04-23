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
 *
 */

#include <tf_test.hpp>

TEST_CASE(stridedslice_masks_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 10, 3, 3}});
    // add literals for starts, ends, and strides in tf (NHWC format)
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 1, 1, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{0, 0, 0, 0});
    mm->add_literal(migraphx::shape{migraphx::shape::int32_type, {4}},
                    std::vector<int>{1, 1, 1, 1});

    auto l1 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0, 1, 1, 0}}, {"ends", {1, 3, 3, 10}}, {"axes", {0, 1, 2, 3}}}),
        l1);
    auto l3 =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), l2);
    mm->add_return({l3});
    auto prog = parse_tf("stridedslice_masks_test.pb", true);

    EXPECT(p == prog);
}
