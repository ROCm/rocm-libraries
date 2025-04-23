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

/* IR for the test case below:
module: "main"
@0 = @literal{0.333333, 0.333333} -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@1 = @literal{0.5, 0.5, 0.5, 0.5} -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@2 = @literal{0, 2, 4, 6, 1, 3, 5, 7} -> int32_type, {4, 1, 1, 2}, {2, 2, 2, 1}
X = @param:X -> float_type, {1, 1, 2, 4}, {8, 8, 4, 1}
@4 = @literal{1, 1, 0.6, 0.5} -> float_type, {4}, {1}
@5 = undefined -> float_type, {}, {}
@6 = reshape[dims={8}](X) -> float_type, {8}, {1}
@7 = gather[axis=0](@6,@2) -> float_type, {4, 1, 1, 2}, {2, 2, 2, 1}
@8 = slice[axes={0},starts={0},ends={2}](@7) -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@9 = slice[axes={0},starts={2},ends={4}](@7) -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@10 = sub(@9,@8) -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@11 = mul(@10,@1) -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@12 = add(@11,@8) -> float_type, {2, 1, 1, 2}, {2, 2, 2, 1}
@13 = slice[axes={0},starts={0},ends={1}](@12) -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@14 = slice[axes={0},starts={1},ends={2}](@12) -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@15 = sub(@14,@13) -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@16 = mul(@15,@0) -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@17 = add(@16,@13) -> float_type, {1, 1, 1, 2}, {2, 2, 2, 1}
@18 = @return(@17)
*/

TEST_CASE(resize_downsample_linear_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    std::vector<float> ds = {1, 1, 0.6, 0.5};
    mm->add_literal(migraphx::literal(ss, ds));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto x = mm->add_parameter("X", sx);

    migraphx::shape s_ind{migraphx::shape::int32_type, {4, 1, 1, 2}};
    std::vector<int> d_ind = {0, 2, 4, 6, 1, 3, 5, 7};
    auto l_ind             = mm->add_literal(migraphx::literal(s_ind, d_ind));

    migraphx::shape s2{migraphx::shape::float_type, {2, 1, 1, 2}};
    std::vector<float> d2(4, 0.5f);
    auto l2 = mm->add_literal(migraphx::literal(s2, d2));

    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 1, 2}};
    std::vector<float> d1(2, 1.0f / 3.0f);
    auto l1 = mm->add_literal(migraphx::literal(s1, d1));

    mm->add_instruction(migraphx::make_op("undefined"));

    auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), x);
    auto data  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, l_ind);
    auto slc20 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), data);
    auto slc21 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), data);
    auto diff2 = mm->add_instruction(migraphx::make_op("sub"), slc21, slc20);
    auto mul2  = mm->add_instruction(migraphx::make_op("mul"), diff2, l2);
    auto add2  = mm->add_instruction(migraphx::make_op("add"), mul2, slc20);

    auto slc10 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add2);
    auto slc11 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add2);
    auto diff1 = mm->add_instruction(migraphx::make_op("sub"), slc11, slc10);
    auto mul1  = mm->add_instruction(migraphx::make_op("mul"), diff1, l1);
    auto add1  = mm->add_instruction(migraphx::make_op("add"), mul1, slc10);
    mm->add_return({add1});

    auto prog = read_onnx("resize_downsample_linear_test.onnx");
    EXPECT(p == prog);
}
