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

#include <onnx_test.hpp>

TEST_CASE(sum_type_test)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l_bool   = mm->add_literal({migraphx::shape{migraphx::shape::bool_type, {2}}, {1, 0}});
    auto l_int8   = mm->add_literal({migraphx::shape{migraphx::shape::int8_type, {2}}, {1, 1}});
    auto l_uint8  = mm->add_literal({migraphx::shape{migraphx::shape::uint8_type, {2}}, {1, 1}});
    auto l_uint16 = mm->add_literal({migraphx::shape{migraphx::shape::uint16_type, {2}}, {1, 1}});
    auto l_uint32 = mm->add_literal({migraphx::shape{migraphx::shape::uint32_type, {2}}, {1, 1}});
    auto l_uint64 = mm->add_literal({migraphx::shape{migraphx::shape::uint64_type, {2}}, {1, 1}});
    auto l_double = mm->add_literal({migraphx::shape{migraphx::shape::double_type, {2}}, {1, 1}});
    auto l_raw  = mm->add_literal({migraphx::shape{migraphx::shape::double_type, {2}}, {1.5, 2.0}});
    auto o_bool = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_bool);
    auto o_int8 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_int8);
    auto o_uint8 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint8);
    auto o_uint16 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint16);
    auto o_uint32 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint32);
    auto o_uint64 = mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::double_type)}}),
        l_uint64);
    auto s0 = mm->add_instruction(migraphx::make_op("add"), o_bool, o_int8);
    auto s1 = mm->add_instruction(migraphx::make_op("add"), s0, o_uint8);
    auto s2 = mm->add_instruction(migraphx::make_op("add"), s1, o_uint16);
    auto s3 = mm->add_instruction(migraphx::make_op("add"), s2, o_uint32);
    auto s4 = mm->add_instruction(migraphx::make_op("add"), s3, o_uint64);
    auto s5 = mm->add_instruction(migraphx::make_op("add"), s4, l_double);
    auto s6 = mm->add_instruction(migraphx::make_op("add"), s5, l_raw);
    mm->add_return({s6});

    auto prog = read_onnx("sum_type_test.onnx");

    EXPECT(p == prog);
}
