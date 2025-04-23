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

TEST_CASE(split_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    std::vector<int64_t> axes{0, 1};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 30}});
    mm->add_literal(3); // num_splits
    mm->add_literal(1); // split axis
    mm->add_literal(1); // concat axis
    mm->add_literal(1); // concat axis
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 0}}, {"ends", {5, 10}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 10}}, {"ends", {5, 20}}}), l0);
    auto l3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", axes}, {"starts", {0, 20}}, {"ends", {5, 30}}}), l0);
    auto l4 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l1, l2);
    auto l5 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l3);
    mm->add_return({l4, l5});
    auto prog = parse_tf("split_test.pb", false);

    EXPECT(p == prog);
}
