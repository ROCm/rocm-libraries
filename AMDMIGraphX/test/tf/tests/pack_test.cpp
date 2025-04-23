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

TEST_CASE(pack_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2}});
    auto l2  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2}});
    std::vector<migraphx::instruction_ref> args{l0, l1, l2};
    std::vector<migraphx::instruction_ref> unsqueezed_args;
    int64_t axis = 1;

    std::transform(
        args.begin(),
        args.end(),
        std::back_inserter(unsqueezed_args),
        [&](migraphx::instruction_ref arg) {
            return mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {axis}}}), arg);
        });
    mm->add_instruction(migraphx::make_op("concat", {{"axis", static_cast<int>(axis)}}),
                        unsqueezed_args);
    auto prog = optimize_tf("pack_test.pb", false);

    EXPECT(p == prog);
}
